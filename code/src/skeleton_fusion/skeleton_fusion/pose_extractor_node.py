import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
import mediapipe as mp

# Detector interface and implementation
from .pose_detectors.mediapipe_detector import MediapipePoseDetector
# Annotator for 2D overlay
from .annotators.pose_annotator import PoseAnnotator
from statistics import mean, stdev

class ImagePoseNode(Node):
    def __init__(self):
        super().__init__('pose_extractor_node')

        # Declare parameters for a single camera
        self.declare_parameter('camera_name', 'cam00')
        self.declare_parameter('color_topic', '/cam00/camera/color/image_raw')
        self.declare_parameter('annotated_topic', '/cam00/camera/color/annotated_image')
        self.declare_parameter('depth_topic', '/cam00/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/cam00/depth/camera_info')
        self.declare_parameter('markerArray_output_topic', '/cam00/pose_3d')
        self.declare_parameter('marker_frame_id', 'cam00_color_optical_frame')

        # Get parameters
        self.cam_name       = self.get_parameter('camera_name').get_parameter_value().string_value
        self.color_topic    = self.get_parameter('color_topic').get_parameter_value().string_value
        self.annotated_topic= self.get_parameter('annotated_topic').get_parameter_value().string_value
        self.depth_topic    = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.caminfo_topic  = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.marker_topic   = self.get_parameter('markerArray_output_topic').get_parameter_value().string_value
        self.marker_frame_id = self.get_parameter('marker_frame_id').get_parameter_value().string_value
        # Initialize bridge, detector, annotator
        self.bridge    = CvBridge()
        self.detector  = MediapipePoseDetector()
        self.annotator = PoseAnnotator()

        # Storage for camera model
        self.cam_model = None

        # Single-use CameraInfo subscriber
        self.info_sub = self.create_subscription(
            CameraInfo,
            self.caminfo_topic,
            self.caminfo_callback,
            10
        )

        # Synchronize color and depth
        color_sub = Subscriber(self, Image, self.color_topic)
        depth_sub = Subscriber(self, Image, self.depth_topic)
        self.sync = ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=10,
            slop=0.05
        )
        self.sync.registerCallback(self.sync_callback)

        # Publishers
        self.img_pub    = self.create_publisher(Image, self.annotated_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        self.get_logger().info(
            f'Node para cámara {self.cam_name}: color={self.color_topic}, depth={self.depth_topic}, caminfo={self.caminfo_topic}, annotated={self.annotated_topic}'
        )

    def caminfo_callback(self, msg: CameraInfo):
        model = PinholeCameraModel()
        model.fromCameraInfo(msg)
        self.cam_model = model
        # Destroy this subscription after first use
        self.destroy_subscription(self.info_sub)
        self.get_logger().info(f'Camera model for {self.cam_name} initialized')

    def sync_callback(self, color_msg: Image, depth_msg: Image):
        # Ensure camera model is ready
        if self.cam_model is None:
            self.get_logger().warn('CameraInfo not received yet')
            return

        # Convert images
        try:
            cv_color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        # 2D pose inference
        rgb = cv2.cvtColor(cv_color, cv2.COLOR_BGR2RGB)
        landmarks = self.detector.process(rgb)

        # Annotate 2D and publish
        annotated = self.annotator.annotate(cv_color, landmarks)
        try:
            out_img = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out_img.header = color_msg.header
            self.img_pub.publish(out_img)
        except Exception as e:
            self.get_logger().error(f'Error publishing annotated image: {e}')

        # Build and publish 3D skeleton
        markers = MarkerArray()
        lifetime = Duration(sec=0, nanosec=100_000_000)
        frame_id = self.marker_frame_id
        stamp = self.get_clock().now().to_msg()

        if landmarks:
            lm_list = landmarks.landmark
            landmarks_3d = {}
            z_values = []

            # Obtener todos los puntos 3D válidos
            for idx, lm in enumerate(lm_list):
                u, v = int(lm.x * depth.shape[1]), int(lm.y * depth.shape[0])
                z = self.get_depth(u, v, depth)
                if z is None:
                    continue
                ray = self.cam_model.projectPixelTo3dRay((u, v))
                pt = [c * z for c in ray]
                landmarks_3d[idx] = pt
                z_values.append(pt[2])

            # Calcular límites de profundidad válidos
            if len(z_values) >= 5:
                z_mean = mean(z_values)
                z_std = stdev(z_values)
                z_min = z_mean - 2 * z_std
                z_max = z_mean + 2 * z_std
            else:
                z_min, z_max = 0.5, 2.5  # fallback

            # Dibujar conexiones solo si ambos extremos están dentro del rango
            for idx, (i, j) in enumerate(mp.solutions.pose.POSE_CONNECTIONS):
                pt1 = landmarks_3d.get(i)
                pt2 = landmarks_3d.get(j)
                if pt1 is None or pt2 is None:
                    continue
                if not (z_min <= pt1[2] <= z_max) or not (z_min <= pt2[2] <= z_max):
                    continue

                m = Marker()
                m.header.frame_id = frame_id
                m.header.stamp = stamp
                m.ns = f'{self.cam_name}_skeleton'
                m.id = idx
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.scale.x = 0.02
                m.color = ColorRGBA(r=0.1, g=0.8, b=0.1, a=1.0)
                m.lifetime = lifetime
                m.points = [Point(x=pt1[0], y=pt1[1], z=pt1[2]), Point(x=pt2[0], y=pt2[1], z=pt2[2])]
                markers.markers.append(m)

            # Dibujar solo los landmarks dentro del rango permitido
            for idx, pt in landmarks_3d.items():
                if not (z_min <= pt[2] <= z_max):
                    continue
                m = Marker()
                m.header.frame_id = frame_id
                m.header.stamp = stamp
                m.ns = f'{self.cam_name}_joints'
                m.id = 1000 + idx
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.scale.x = 0.04
                m.scale.y = 0.04
                m.scale.z = 0.04
                m.color = ColorRGBA(r=0.9, g=0.1, b=0.1, a=1.0)
                m.pose.position = Point(x=pt[0], y=pt[1], z=pt[2])
                m.lifetime = lifetime
                markers.markers.append(m)

        self.marker_pub.publish(markers)

    def get_depth(self, u: int, v: int, depth: Image, max_depth: float = 2.5, min_depth: float = 0.5) -> float:
        """Get depth value at pixel (u, v) in the depth image."""
        if not self.valid_pixel(u, v, depth.shape):
            self.get_logger().warn(f'Invalid pixel coordinates: ({u}, {v})')
            return None
        try:
            depth_value =  depth[v, u] * 0.001  # Convert to meters
            if depth_value < min_depth or depth_value > max_depth:
                self.get_logger().warn(f'Depth out of range: {depth_value} at ({u}, {v})')
                return None
            return depth_value
        except Exception as e:
            self.get_logger().error(f'Error getting depth: {e}')
            return None
    
    def valid_pixel(self, u: int, v: int, shape: tuple) -> bool:
        """Check if pixel coordinates (u, v) are valid for the given image shape."""
        height, width = shape[:2]
        return 0 <= u < width and 0 <= v < height
    
def main(args=None):
    rclpy.init(args=args)
    node = ImagePoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
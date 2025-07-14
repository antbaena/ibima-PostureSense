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
from pose_detectors.mediapipe_detector import MediapipePoseDetector
# Annotator for 2D overlay
from annotators.pose_annotator import PoseAnnotator

class ImagePoseNode(Node):
    def __init__(self):
        super().__init__('pose_extractor_node')

        # Declare parameters for a single camera
        self.declare_parameter('camera_name', 'cam00')
        self.declare_parameter('color_topic', '/cam00/camera/color/image_raw')
        self.declare_parameter('annotated_topic', '/cam00/camera/color/annotated_image')
        self.declare_parameter('depth_topic', '/cam00/depth/image_raw')
        self.declare_parameter('caminfo_topic', '/cam00/depth/camera_info')

        # Get parameters
        self.cam_name       = self.get_parameter('camera_name').get_parameter_value().string_value
        self.color_topic    = self.get_parameter('color_topic').get_parameter_value().string_value
        self.annotated_topic= self.get_parameter('annotated_topic').get_parameter_value().string_value
        self.depth_topic    = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.caminfo_topic  = self.get_parameter('caminfo_topic').get_parameter_value().string_value

        # Initialize bridge, detector, annotator
        self.bridge    = CvBridge()
        self.detector  = MediapipePoseDetector()
        self.annotator = PoseAnnotator()

        # Storage for camera model
        self.cam_model = None

        # Single-use CameraInfo subscriber
        self.create_subscription(
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
        self.marker_pub = self.create_publisher(MarkerArray, f'/{self.cam_name}/pose_3d', 10)

        self.get_logger().info(
            f'Node para cámara {self.cam_name}: color={self.color_topic}, depth={self.depth_topic}, caminfo={self.caminfo_topic}, annotated={self.annotated_topic}'
        )

    def caminfo_callback(self, msg: CameraInfo):
        model = PinholeCameraModel()
        model.fromCameraInfo(msg)
        self.cam_model = model
        # Destroy this subscription after first use
        self.destroy_subscription(self.caminfo_callback)
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
        lifetime = Duration(sec=0, nanosec=500_000_000)
        frame_id = color_msg.header.frame_id

        if landmarks:
            lm_list = landmarks.landmark
            for idx, (i, j) in enumerate(mp.solutions.pose.POSE_CONNECTIONS):
                u1, v1 = int(lm_list[i].x * depth.shape[1]), int(lm_list[i].y * depth.shape[0])
                u2, v2 = int(lm_list[j].x * depth.shape[1]), int(lm_list[j].y * depth.shape[0])
                z1 = depth[v1, u1] * 0.001
                z2 = depth[v2, u2] * 0.001
                ray1 = self.cam_model.projectPixelTo3dRay((u1, v1))
                ray2 = self.cam_model.projectPixelTo3dRay((u2, v2))
                pt1 = [c * z1 for c in ray1]
                pt2 = [c * z2 for c in ray2]

                m = Marker()
                m.header.frame_id = frame_id
                m.header.stamp = color_msg.header.stamp
                m.ns = f'{self.cam_name}_skeleton'
                m.id = idx
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.scale.x = 0.02
                m.color = ColorRGBA(r=0.1, g=0.8, b=0.1, a=1.0)
                m.lifetime = lifetime
                m.points = [Point(x=pt1[0], y=pt1[1], z=pt1[2]), Point(x=pt2[0], y=pt2[1], z=pt2[2])]
                markers.markers.append(m)

            for idx, lm in enumerate(lm_list):
                u, v = int(lm.x * depth.shape[1]), int(lm.y * depth.shape[0])
                z = depth[v, u] * 0.001
                ray = self.cam_model.projectPixelTo3dRay((u, v))
                pt = [c * z for c in ray]

                m = Marker()
                m.header.frame_id = frame_id
                m.header.stamp = color_msg.header.stamp
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


def main(args=None):
    rclpy.init(args=args)
    node = ImagePoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from std_msgs.msg import Float64
import tf2_ros
import tf2_geometry_msgs
from builtin_interfaces.msg import Time

from message_filters import Subscriber, ApproximateTimeSynchronizer

import math
import numpy as np

class SkeletonFusionNode(Node):
    def __init__(self):
        super().__init__('skeleton_fusion_node')

        # Params
        self.declare_parameter('camera_names', ['cam00/camera_00', 'cam00/camera_01', 'cam3', 'cam4'])
        self.declare_parameter('target_frame', 'cam1')

        self.camera_names = self.get_parameter('camera_names').get_parameter_value().string_array_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers
        self.fused_pub = self.create_publisher(MarkerArray, '/fused_skeleton', 10)
        self.error_pub = self.create_publisher(Float64, '/skeleton_fusion/error_metric', 10)

        # Subscribers
        self.subs = [Subscriber(self, MarkerArray, f'/{name}/pose_3d') for name in self.camera_names]
        self.sync = ApproximateTimeSynchronizer(self.subs, queue_size=5, slop=0.1, allow_headerless=True)
        self.sync.registerCallback(self.sync_callback)

    def sync_callback(self, *marker_arrays):
        try:
            # Transform all marker arrays to target_frame
            all_points_per_cam = []

            for name, marker_array in zip(self.camera_names, marker_arrays):
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    marker_array.markers[0].header.frame_id,
                    rclpy.time.Time()
                )

                cam_points = []
                for m in marker_array.markers:
                    if len(m.points) != 2:
                        continue
                    # Transform each point
                    p1 = tf2_geometry_msgs.do_transform_point(m.points[0], transform)
                    p2 = tf2_geometry_msgs.do_transform_point(m.points[1], transform)
                    cam_points.append(((p1.point.x + p2.point.x)/2.0,
                                       (p1.point.y + p2.point.y)/2.0,
                                       (p1.point.z + p2.point.z)/2.0))  # centro de hueso

                all_points_per_cam.append(np.array(cam_points))

        except Exception as ex:
            self.get_logger().warn(f"TF transform failed: {str(ex)}")
            return

        # Simple fusion: promedio de puntos
        fused_points = np.mean(all_points_per_cam, axis=0)

        # Error metric: distancia media entre estimaciones
        errors = []
        for cam_points in all_points_per_cam:
            if cam_points.shape != fused_points.shape:
                continue
            dists = np.linalg.norm(fused_points - cam_points, axis=1)
            errors.append(np.mean(dists))
        global_error = float(np.mean(errors)) if errors else 0.0

        # Publish fused MarkerArray
        fused_msg = MarkerArray()
        for i, (x, y, z) in enumerate(fused_points):
            marker = Marker()
            marker.header.frame_id = self.target_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "fused_skeleton"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = marker.scale.y = marker.scale.z = 0.05
            marker.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            fused_msg.markers.append(marker)

        self.fused_pub.publish(fused_msg)

        # Publish error metric
        err_msg = Float64()
        err_msg.data = global_error
        self.error_pub.publish(err_msg)

        self.get_logger().info(f"Fusion complete - error: {global_error:.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = SkeletonFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

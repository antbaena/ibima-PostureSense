import rclpy
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

from extrinsic_calibrator_interfaces.srv import BroadcastPoses 
from extrinsic_calibrator_interfaces.msg import CameraTransform

class StaticTFBroadcaster(Node):

    def __init__(self):
        super().__init__('static_tf_broadcaster')
        self.broadcaster = StaticTransformBroadcaster(self)
        self.srv = self.create_service(BroadcastPoses, 'broadcast_static_tfs', self.handle_broadcast)
        self.get_logger().info('Static TF Broadcaster Node Initialized.')

    def handle_broadcast(self, request, response):
        if not request.poses:
            response.success = False
            response.message = "No transforms provided"
            return response

        try:
            transforms = []
            for pose in request.poses:
                tf = TransformStamped()
                tf.header.stamp = self.get_clock().now().to_msg()
                tf.header.frame_id = pose.parent_frame
                tf.child_frame_id = pose.child_frame
                tf.transform.translation = pose.translation
                tf.transform.rotation = pose.rotation
                transforms.append(tf)

            self.broadcaster.sendTransform(transforms)

            response.success = True
            response.message = f"Broadcasted {len(transforms)} static transforms"
            return response

        except Exception as e:
            self.get_logger().error(f"Exception: {e}")
            response.success = False
            response.message = f"Error: {str(e)}"
            return response


def main(args=None):
    rclpy.init(args=args)
    node = StaticTFBroadcaster()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

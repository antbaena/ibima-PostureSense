import tf2_ros
from geometry_msgs.msg import TransformStamped

class StaticTfPublisher:
    def __init__(self, node):
        self._broadcaster = tf2_ros.StaticTransformBroadcaster(node)

    def send(self, transforms: list[TransformStamped]):
        if transforms:
            self._broadcaster.sendTransform(transforms)

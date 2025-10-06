from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    config_file = PathJoinSubstitution([
        FindPackageShare("skeleton_fusion"),
        "config",
        "camera.yaml"
    ])

    return LaunchDescription([
        Node(
            package="skeleton_fusion",
            executable="pose_extractor_node",
            name="cam00_00",
            output="screen",
            parameters=[config_file],
        ),
        Node(
            package="skeleton_fusion",
            executable="pose_extractor_node",
            name="cam00_01",
            output="screen",
            parameters=[config_file],
        ),
        Node(
            package="skeleton_fusion",
            executable="pose_extractor_node",
            name="cam01_02",
            output="screen",
            parameters=[config_file],
        ),
        Node(
            package="skeleton_fusion",
            executable="pose_extractor_node",
            name="cam02_03",
            output="screen",
            parameters=[config_file],
        ),
    ])

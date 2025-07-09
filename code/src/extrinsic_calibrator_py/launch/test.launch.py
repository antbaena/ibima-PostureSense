from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    config_file = PathJoinSubstitution([
        FindPackageShare("extrinsic_calibrator_py"),
        "config",
        "params.yaml"
    ])

    return LaunchDescription([
        Node(
            package="extrinsic_calibrator_py",
            executable="extrinsic_calibrator",
            name="detector_aruco_node",
            output="screen",
            parameters=[config_file],
        )
    ])

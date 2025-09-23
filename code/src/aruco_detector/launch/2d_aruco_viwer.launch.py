# launch/aruco_yaml_launch.py
import os
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('aruco_detector')
    param_file = os.path.join(pkg_share, 'config', 'aruco_params.yaml')
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter file not found: {param_file}")
    return LaunchDescription([
        Node(
            package='aruco_detector',
            executable='aruco_detector_node',
            name='aruco_detector_node',
            output='screen',
            parameters=[param_file],
        )
    ])

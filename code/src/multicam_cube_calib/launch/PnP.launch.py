# launch/aruco_yaml_launch.py
import os
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='multicam_cube_calib',
            executable='synchronized_pair_PnP_publisher_node',
            name='cam_00_01_PnP_publisher',
            parameters=[
                {'cameras': ['cam00/camera_00', 'cam01/camera_02']}
            ],
            output='screen',
            prefix="xterm -hold -e",
            emulate_tty=True,
            
        ),
        Node(
            package='multicam_cube_calib',
            executable='synchronized_pair_PnP_publisher_node',
            name='cam_00_01_PnP_publisher',
            parameters=[
                {'cameras': ['cam00/camera_00', 'cam00/camera_01']}
            ],
            output='screen',
            prefix="xterm -hold -e",
            emulate_tty=True,
        ),
        Node(
            package='multicam_cube_calib',
            executable='synchronized_pair_PnP_publisher_node',
            name='cam_01_02_PnP_publisher',
            parameters=[
                {'cameras': ['cam01/camera_02', 'cam02/camera_03']}
            ],
            output='screen',
            prefix="xterm -hold -e",
            emulate_tty=True,
        ),
        Node(
            package='multicam_cube_calib',
            executable='extrinsics_optimizer_node',
            name='extrinsics_optimizer_node',
            parameters=[
                { 'root_frame_id': 'cam00/camera_00',
                    'camera_names': ['cam00/camera_00', 'cam00/camera_01'],
                      'link_names' : ['camera_00_color_optical_frame', 'camera_01_color_optical_frame'],
                }
            ],
            output='screen',
            prefix="xterm -hold -e",
            emulate_tty=True,
        ),
        Node(
            package='multicam_cube_calib',
            executable='extrinsics_optimizer_node',
            name='extrinsics_optimizer_node',
            parameters=[
                { 'root_frame_id': 'cam00/camera_00',
                    'camera_names': ['cam00/camera_00', 'cam01/camera_02'],
                        'link_names' : ['camera_00_color_optical_frame', 'camera_02_color_optical_frame'],
                }
            ],
            output='screen',
            prefix="xterm -hold -e",
            emulate_tty=True,
        ),
        Node(
            package='multicam_cube_calib',
            executable='extrinsics_optimizer_node',
            name='extrinsics_optimizer_node',
            parameters=[
                { 'root_frame_id': 'cam01/camera_02',
                    'camera_names': ['cam01/camera_02', 'cam02/camera_03'],
                        'link_names' : ['camera_02_color_optical_frame', 'camera_03_color_optical_frame'],
                }
            ],
            output='screen',
            prefix="xterm -hold -e",
            emulate_tty=True,
        ),
        
    ])

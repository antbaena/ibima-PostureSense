from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    ld = LaunchDescription()
    cameras = ['/cam00/camera_00', '/cam00/camera_01', '/cam01/camera_02', '/cam02/camera_03']

    for idx, cam in enumerate(cameras, start=1):
        in_arg = f'input_topic_{idx}'
        out_arg = f'output_topic_{idx}'

        ld.add_action(DeclareLaunchArgument(
            in_arg,
            default_value=f'{cam}/color/image_raw/compressed',
            description=f'Compressed input topic for camera {idx}'
        ))
        ld.add_action(DeclareLaunchArgument(
            out_arg,
            default_value=f'{cam}/color/image_raw/decompressed',
            description=f'Raw output topic for camera {idx}'
        ))

        ld.add_action(Node(
            package='gemini_utils',
            executable='uncompress_node',
            name=f'uncompress_node_{idx}',
            output='screen',
            parameters=[{
                'input_topic': LaunchConfiguration(in_arg),
                'output_topic': LaunchConfiguration(out_arg),
                'queue_size': 10
            }]
        ))
    return ld
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os
from launch_ros.actions import PushRosNamespace


def generate_launch_description():
    # Include launch files
    package_dir = get_package_share_directory('orbbec_camera')
    launch_file_dir = os.path.join(package_dir, 'launch')

    common_args = {
        'device_num': '2',
        'sync_mode': 'standalone',
        'color_fps': '30',
        'depth_fps': '30',
        'color_width': '640',
        'color_height': '480',
        'depth_width': '640',
        'depth_height': '480',
        'enable_gyro': 'true',
        'enable_accel': 'true',
        'depth_registration': 'true',

        # Compresión + menos ancho de banda
        'color_format': 'MJPG',
        'depth_format': 'Y16',

        # Desactiva streams extra
        'enable_point_cloud': 'false',
        'enable_colored_point_cloud': 'false',
        'enable_left_ir': 'false',
        'enable_right_ir': 'false',

        # Filtros desactivados
        'enable_temporal_filter': 'false',
        'enable_spatial_filter': 'false',
        'enable_hole_filling_filter': 'false',
        'enable_threshold_filter': 'false',
        'enable_decimation_filter': 'false',
        'enable_noise_removal_filter': 'true',  # este puede ayudar con ruido sin penalizar mucho

        # Extra rendimiento
        'publish_tf': 'false',
        'enable_3d_reconstruction_mode': 'false',
    }

    launch1_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'gemini_330_series.launch.py')
        ),
        launch_arguments={
            **common_args,
            'sync_mode': 'primary',
            'camera_name': 'camera_00',
            'serial_number': 'CP32942000G0',
        }.items()
    )

    launch2_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'gemini_330_series.launch.py')
        ),
        launch_arguments={
            **common_args,
            'sync_mode': 'secondary',
            'depth_delay_us': '16',
            'color_delay_us': '16',
            'camera_name': 'camera_01',
            'serial_number': 'CP329420002N',
        }.items()
    )

    return LaunchDescription([
        PushRosNamespace('cam00'),
        GroupAction([launch1_include]),
        GroupAction([launch2_include]),
    ])

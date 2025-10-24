from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import PushRosNamespace
import os


def generate_launch_description():
    # Directorio del package y del launch genérico
    package_dir = get_package_share_directory('orbbec_camera')
    launch_file_dir = os.path.join(package_dir, 'launch')

    # Parámetros agrupados (formato similar al dual)
    common_args = {
        'device_num': '1',
        'sync_mode': 'secondary',

        # Frecuencias y resoluciones
        'color_fps': '30', #15
        'depth_fps': '30',
        'color_width': '640', #1280
        'color_height': '480', #800
        'depth_width': '640', 
        'depth_height': '480',

        # IMU y registro
        'enable_gyro': 'true',
        'enable_accel': 'true',
        'depth_registration': 'true',

        # Compresión / formato
        'color_format': 'MJPG',
        'depth_format': 'Y16',

        # Streams extra desactivados
        'enable_point_cloud': 'false',
        'enable_colored_point_cloud': 'false',
        'enable_left_ir': 'false',
        'enable_right_ir': 'false',

        # Filtros
        'enable_temporal_filter': 'false',
        'enable_spatial_filter': 'false',
        'enable_hole_filling_filter': 'false',
        'enable_threshold_filter': 'false',
        'enable_decimation_filter': 'false',
        'enable_noise_removal_filter': 'true',

        # Extra rendimiento
        'publish_tf': 'false',
        'enable_3d_reconstruction_mode': 'false',

        # Identidad de la cámara
        'camera_name': 'camera_02',
        'serial_number': 'CPCS2530001K', #CP32942000RH

        # Delays (no relevantes en standalone; se dejan a 0 para simetría)
        'depth_delay_us': '32', #48
        'color_delay_us': '32', #48
    }

    single_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'gemini_330_series.launch.py')
        ),
        launch_arguments=common_args.items(),
    )

    return LaunchDescription([
        # Namespace para aislar tópicos/frames de esta cámara
        PushRosNamespace('cam01'), #cam02
        GroupAction([single_include]),
    ])
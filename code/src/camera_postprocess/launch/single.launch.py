from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import PushRosNamespace, Node
import os


def generate_launch_description():
    # ── Launch arguments ────────────────────────────────────────────
    namespace_arg = DeclareLaunchArgument('namespace', default_value='cam01',
                                          description='ROS namespace for this host')
    camera_name_arg = DeclareLaunchArgument('camera_name', default_value='camera_02',
                                             description='Camera node name')
    serial_arg = DeclareLaunchArgument('serial_number', default_value='CPCS2530001K',
                                       description='Camera serial number')
    sync_mode_arg = DeclareLaunchArgument('sync_mode', default_value='secondary',
                                          description='Sync mode: standalone/primary/secondary')
    color_fps_arg = DeclareLaunchArgument('color_fps', default_value='30')
    depth_fps_arg = DeclareLaunchArgument('depth_fps', default_value='30')
    # 1280x720 chosen for offline MediaPipe Pose at ~4 m (subject ~432 px tall).
    # Verified stable on Gemini 330 + RPi 5: 30 Hz, 4.5 MB/s MJPG, 0 frame drops.
    color_width_arg = DeclareLaunchArgument('color_width', default_value='1280')
    color_height_arg = DeclareLaunchArgument('color_height', default_value='720')
    depth_width_arg = DeclareLaunchArgument('depth_width', default_value='640')
    depth_height_arg = DeclareLaunchArgument('depth_height', default_value='480')
    preview_fps_arg = DeclareLaunchArgument('preview_fps', default_value='5.0')
    preview_scale_arg = DeclareLaunchArgument('preview_scale', default_value='0.5')
    preview_quality_arg = DeclareLaunchArgument('preview_quality', default_value='50')

    # ── Orbbec driver ───────────────────────────────────────────────
    package_dir = get_package_share_directory('orbbec_camera')
    launch_file_dir = os.path.join(package_dir, 'launch')

    common_args = {
        'device_num': '1',
        'sync_mode': LaunchConfiguration('sync_mode'),
        # FPS and resolution
        'color_fps': LaunchConfiguration('color_fps'),
        'depth_fps': LaunchConfiguration('depth_fps'),
        'color_width': LaunchConfiguration('color_width'),
        'color_height': LaunchConfiguration('color_height'),
        'depth_width': LaunchConfiguration('depth_width'),
        'depth_height': LaunchConfiguration('depth_height'),
        # IMU and registration
        'enable_gyro': 'true',
        'enable_accel': 'true',
        'depth_registration': 'true',
        # Compression
        'color_format': 'MJPG',
        'depth_format': 'Y16',
        # Disable extras
        'enable_point_cloud': 'false',
        'enable_colored_point_cloud': 'false',
        'enable_left_ir': 'false',
        'enable_right_ir': 'false',
        # Filters
        'enable_temporal_filter': 'false',
        'enable_spatial_filter': 'false',
        'enable_hole_filling_filter': 'false',
        'enable_threshold_filter': 'false',
        'enable_decimation_filter': 'false',
        'enable_noise_removal_filter': 'true',
        # QoS — sensor_data for all camera streams
        'color_qos': 'sensor_data',
        'color_camera_info_qos': 'sensor_data',
        'depth_qos': 'sensor_data',
        'depth_camera_info_qos': 'sensor_data',
        # Performance
        'publish_tf': 'false',
        'enable_3d_reconstruction_mode': 'false',
        # Camera identity
        'camera_name': LaunchConfiguration('camera_name'),
        'serial_number': LaunchConfiguration('serial_number'),
        # Delays
        'depth_delay_us': '32',
        'color_delay_us': '32',
    }

    single_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'gemini_330_series.launch.py')
        ),
        launch_arguments={k: (v if isinstance(v, str) else v) for k, v in common_args.items()}.items(),
    )

    ns = LaunchConfiguration('namespace')
    cam = LaunchConfiguration('camera_name')

    # ── Preview republisher ─────────────────────────────────────────
    preview_node = Node(
        package='camera_postprocess',
        executable='preview_republisher',
        name='preview_republisher',
        output='log',
        parameters=[{
            'input_topic': ['/', ns, '/', cam, '/color/image_raw/compressed'],
            'output_topic': ['/', ns, '/', cam, '/color/preview/compressed'],
            'target_fps': LaunchConfiguration('preview_fps'),
            'scale_factor': LaunchConfiguration('preview_scale'),
            'jpeg_quality': LaunchConfiguration('preview_quality'),
        }],
    )

    return LaunchDescription([
        # Launch arguments
        namespace_arg,
        camera_name_arg,
        serial_arg,
        sync_mode_arg,
        color_fps_arg,
        depth_fps_arg,
        color_width_arg,
        color_height_arg,
        depth_width_arg,
        depth_height_arg,
        preview_fps_arg,
        preview_scale_arg,
        preview_quality_arg,
        # Camera driver under namespace
        PushRosNamespace(ns),
        GroupAction([single_include]),
        # Preview republisher
        preview_node,
    ])
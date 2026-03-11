from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os
from launch_ros.actions import PushRosNamespace, Node


def generate_launch_description():
    # ── Launch arguments ────────────────────────────────────────────
    namespace_arg = DeclareLaunchArgument('namespace', default_value='cam00',
                                          description='ROS namespace for this host')
    serial_primary_arg = DeclareLaunchArgument('serial_primary', default_value='CP32942000G0',
                                               description='Serial number of primary camera')
    serial_secondary_arg = DeclareLaunchArgument('serial_secondary', default_value='CP329420002N',
                                                  description='Serial number of secondary camera')
    color_fps_arg = DeclareLaunchArgument('color_fps', default_value='30',
                                          description='Color stream FPS')
    depth_fps_arg = DeclareLaunchArgument('depth_fps', default_value='30',
                                          description='Depth stream FPS')
    color_width_arg = DeclareLaunchArgument('color_width', default_value='640')
    color_height_arg = DeclareLaunchArgument('color_height', default_value='480')
    depth_width_arg = DeclareLaunchArgument('depth_width', default_value='640')
    depth_height_arg = DeclareLaunchArgument('depth_height', default_value='480')
    preview_fps_arg = DeclareLaunchArgument('preview_fps', default_value='5.0',
                                            description='Preview republisher target FPS')
    preview_scale_arg = DeclareLaunchArgument('preview_scale', default_value='0.5',
                                              description='Preview downscale factor')
    preview_quality_arg = DeclareLaunchArgument('preview_quality', default_value='50',
                                                description='Preview JPEG quality [1-100]')
    enable_preview_arg = DeclareLaunchArgument('enable_preview', default_value='true',
                                               description='Launch preview republisher nodes')

    # ── Orbbec driver paths ─────────────────────────────────────────
    package_dir = get_package_share_directory('orbbec_camera')
    launch_file_dir = os.path.join(package_dir, 'launch')

    common_args = {
        'device_num': '2',
        'color_fps': LaunchConfiguration('color_fps'),
        'depth_fps': LaunchConfiguration('depth_fps'),
        'color_width': LaunchConfiguration('color_width'),
        'color_height': LaunchConfiguration('color_height'),
        'depth_width': LaunchConfiguration('depth_width'),
        'depth_height': LaunchConfiguration('depth_height'),
        'enable_gyro': 'true',
        'enable_accel': 'true',
        'depth_registration': 'true',
        # Compression + bandwidth
        'color_format': 'MJPG',
        'depth_format': 'Y16',
        # Disable extra streams
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
    }

    # ── Primary camera ──────────────────────────────────────────────
    launch1_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'gemini_330_series.launch.py')
        ),
        launch_arguments={
            **{k: (v if isinstance(v, str) else v) for k, v in common_args.items()},
            'sync_mode': 'primary',
            'camera_name': 'camera_00',
            'serial_number': LaunchConfiguration('serial_primary'),
        }.items()
    )

    # ── Secondary camera ────────────────────────────────────────────
    launch2_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'gemini_330_series.launch.py')
        ),
        launch_arguments={
            **{k: (v if isinstance(v, str) else v) for k, v in common_args.items()},
            'sync_mode': 'secondary',
            'depth_delay_us': '16',
            'color_delay_us': '16',
            'camera_name': 'camera_01',
            'serial_number': LaunchConfiguration('serial_secondary'),
        }.items()
    )

    # ── Preview republisher nodes ───────────────────────────────────
    ns = LaunchConfiguration('namespace')

    preview_cam00 = Node(
        package='camera_postprocess',
        executable='preview_republisher',
        name='preview_republisher_cam00',
        output='log',
        parameters=[{
            'input_topic': ['/', ns, '/camera_00/color/image_raw/compressed'],
            'output_topic': ['/', ns, '/camera_00/color/preview/compressed'],
            'target_fps': LaunchConfiguration('preview_fps'),
            'scale_factor': LaunchConfiguration('preview_scale'),
            'jpeg_quality': LaunchConfiguration('preview_quality'),
        }],
    )

    preview_cam01 = Node(
        package='camera_postprocess',
        executable='preview_republisher',
        name='preview_republisher_cam01',
        output='log',
        parameters=[{
            'input_topic': ['/', ns, '/camera_01/color/image_raw/compressed'],
            'output_topic': ['/', ns, '/camera_01/color/preview/compressed'],
            'target_fps': LaunchConfiguration('preview_fps'),
            'scale_factor': LaunchConfiguration('preview_scale'),
            'jpeg_quality': LaunchConfiguration('preview_quality'),
        }],
    )

    return LaunchDescription([
        # Launch arguments
        namespace_arg,
        serial_primary_arg,
        serial_secondary_arg,
        color_fps_arg,
        depth_fps_arg,
        color_width_arg,
        color_height_arg,
        depth_width_arg,
        depth_height_arg,
        preview_fps_arg,
        preview_scale_arg,
        preview_quality_arg,
        enable_preview_arg,
        # Camera drivers under namespace
        PushRosNamespace(ns),
        GroupAction([launch1_include]),
        GroupAction([launch2_include]),
        # Preview republishers
        preview_cam00,
        preview_cam01,
    ])

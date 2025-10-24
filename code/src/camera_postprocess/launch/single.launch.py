#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import PushRosNamespace


def generate_launch_description():
    # Ruta al package orbbec_camera
    pkg_dir = get_package_share_directory("orbbec_camera")
    launch_dir = os.path.join(pkg_dir, "launch")

    # Declaración de argumentos de lanzamiento
    args = [
        DeclareLaunchArgument(
            "device_num",
            default_value="1",
        ),
        DeclareLaunchArgument(
            "sync_mode",
            default_value="secondary",
        ),
        DeclareLaunchArgument("depth_delay_us", default_value="32"), #48
        DeclareLaunchArgument("color_delay_us", default_value="32"), #48
        DeclareLaunchArgument(
            "color_fps",
            default_value="15",
        ),
        DeclareLaunchArgument(
            "color_width",
            default_value="1280",
        ),
        DeclareLaunchArgument(
            "color_height",
            default_value="800",
        ),
        DeclareLaunchArgument(
            "color_format",
            default_value="MJPG",
        ),
        DeclareLaunchArgument(
            "camera_name",
            default_value="camera_02",
        ),
        DeclareLaunchArgument(
            "serial_number",
            default_value="CPCS2530001K", #CP32942000RH
        ),
        DeclareLaunchArgument(
            "namespace",
            default_value="cam01", #cam02
        ),
        DeclareLaunchArgument(
            "depth_registration",
            default_value="true",
        ),
    ]

    # Incluir el launch genérico de gemini_330_series con los argumentos
    single_cam = PythonLaunchDescriptionSource(
        os.path.join(launch_dir, "gemini_330_series.launch.py")
    )
    include_camera = GroupAction(
        [
            # Se envuelven en un GroupAction para aislar namespaces si se quisiera
            PythonLaunchDescriptionSource  # placeholder, sustituido abajo
        ]
    )

    include_camera = IncludeLaunchDescription(
        single_cam,
        launch_arguments={
            "sync_mode": LaunchConfiguration("sync_mode"),
            "color_fps": LaunchConfiguration("color_fps"),
            "color_width": LaunchConfiguration("color_width"),
            "color_height": LaunchConfiguration("color_height"),
            "color_format": LaunchConfiguration("color_format"),
            "camera_name": LaunchConfiguration("camera_name"),
            "serial_number": LaunchConfiguration("serial_number"),
        }.items(),
    )

    return LaunchDescription(args + [
        GroupAction([
            PushRosNamespace(LaunchConfiguration("namespace")),
            include_camera
        ])
    ])
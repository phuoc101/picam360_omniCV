import argparse
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node

import os
import sys

# from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = LaunchDescription()

    # get path to params file
    dir = os.path.dirname(os.path.abspath(__file__))
    ld.add_action(action=DeclareLaunchArgument(
        'video_device', default_value=TextSubstitution(text='/dev/video2')))
    ld.add_action(action=DeclareLaunchArgument(
        'io_method', default_value=TextSubstitution(text='mmap')))
    ld.add_action(action=DeclareLaunchArgument(
        'frame_id', default_value=TextSubstitution(text='picam360')))
    ld.add_action(action=DeclareLaunchArgument(
        'pixel_format', default_value=TextSubstitution(text='mjpeg')))
    ld.add_action(action=DeclareLaunchArgument(
        'image_width', default_value=TextSubstitution(text='640')))
    ld.add_action(action=DeclareLaunchArgument(
        'image_height', default_value=TextSubstitution(text='480')))
    ld.add_action(action=DeclareLaunchArgument(
        'framerate', default_value=TextSubstitution(text='10.')))
    ld.add_action(action=DeclareLaunchArgument(
        'camera_name', default_value=TextSubstitution(text='picam360')))

    ld.add_action(Node(
        package='usb_cam', executable='usb_cam_node_exe', output='screen',
        name='picam360',
        # namespace=ns,
        parameters=[
            {
                'video_device': LaunchConfiguration('video_device'),
                'io_method': LaunchConfiguration('io_method'),
                'frame_id': LaunchConfiguration('frame_id'),
                'pixel_format': LaunchConfiguration('pixel_format'),
                'image_height': LaunchConfiguration('image_height'),
                'image_width': LaunchConfiguration('image_width'),
                'framerate': LaunchConfiguration('framerate'),
                'camera_name': LaunchConfiguration('camera_name'),
            }],
        remappings=[
            ("__ns", "/picam360")
        ]
    ))
    ld.add_action(Node(
        package='omnicv_ros2', executable='show_img', output='screen',
    ))
    return ld

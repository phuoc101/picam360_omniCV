from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node

import os
import sys

def generate_launch_description():

    video_device_launch_arg = DeclareLaunchArgument(
        'video_device', default_value=TextSubstitution(text='/dev/video2'))
    io_method_launch_arg = DeclareLaunchArgument(
        'io_method', default_value=TextSubstitution(text='mmap'))
    frame_id_launch_arg = DeclareLaunchArgument(
        'frame_id', default_value=TextSubstitution(text='picam360'))
    pixel_format_launch_arg = DeclareLaunchArgument(
        'pixel_format', default_value=TextSubstitution(text='mjpeg'))
    image_width_launch_arg = DeclareLaunchArgument(
        'image_width', default_value=TextSubstitution(text='640'))
    image_height_launch_arg = DeclareLaunchArgument(
        'image_height', default_value=TextSubstitution(text='480'))
    framerate_launch_arg = DeclareLaunchArgument(
        'framerate', default_value=TextSubstitution(text='10.'))
    camera_name_launch_arg = DeclareLaunchArgument(
        'camera_name', default_value=TextSubstitution(text='picam360'))

    usb_cam_node = Node(
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
    )
    show_img_node = Node(
        package='omnicv_ros2', executable='show_img', output='screen',
    )
    return LaunchDescription([
            video_device_launch_arg,
            io_method_launch_arg,
            frame_id_launch_arg,
            pixel_format_launch_arg,
            image_width_launch_arg,
            image_height_launch_arg,
            framerate_launch_arg,
            camera_name_launch_arg,
            usb_cam_node,
            show_img_node,
        ])

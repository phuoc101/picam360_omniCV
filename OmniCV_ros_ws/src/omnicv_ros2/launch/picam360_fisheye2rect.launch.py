from ament_index_python import get_package_share_directory
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
    ld = LaunchDescription()

    ld.add_action(action=DeclareLaunchArgument(
        'image_width', default_value=TextSubstitution(text='640')))
    ld.add_action(action=DeclareLaunchArgument(
        'image_height', default_value=TextSubstitution(text='480')))
    ld.add_action(action=DeclareLaunchArgument(
        'out_width', default_value=TextSubstitution(text='800')))
    ld.add_action(action=DeclareLaunchArgument(
        'out_height', default_value=TextSubstitution(text='400')))

    ld.add_action(action=IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file_path=os.path.join(
            get_package_share_directory('omnicv_ros2'),
            'launch/picam360_read.launch.py')),
    ))
    ld.add_action(Node(
        package='omnicv_ros2', executable='fisheye2rect', output="screen",
        name="picam_rectifier",
        parameters=[
            {'image_width': LaunchConfiguration('image_width')},
            {'image_height': LaunchConfiguration('image_height')},
            {'out_width': LaunchConfiguration('out_width')},
            {'out_height': LaunchConfiguration('out_height')},
            ]
    ))
    return ld

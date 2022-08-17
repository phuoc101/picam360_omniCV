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

    ld.add_action(action=DeclareLaunchArgument(
        'image_width', default_value=TextSubstitution(text='640')))
    ld.add_action(action=DeclareLaunchArgument(
        'image_height', default_value=TextSubstitution(text='480')))

    ld.add_action(action=IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_file_path=os.path.join(
            get_package_share_directory('omnicv_ros2'),
            'launch/picam360_read.launch.py')),
    ))

    ld.add_action(Node(
        package='omnicv_ros2', executable='360view', output="screen",
        name="picam_360viewer"
        ))
    return ld

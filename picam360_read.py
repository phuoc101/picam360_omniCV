import argparse
from launch import LaunchDescription
from launch_ros.actions import Node

import os
import sys

# from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = LaunchDescription()

    parser = argparse.ArgumentParser(description='usb_cam demo')
    parser.add_argument('-n', '--node-name', dest='node_name', type=str,
                        help='name for device', default='usb_cam')

    args, unknown = parser.parse_known_args(sys.argv[4:])

    # usb_cam_dir = get_package_share_directory('usb_cam')

    # get path to params file
    params_path = 'picam360_params.yaml'

    node_name = args.node_name

    print(params_path)
    ld.add_action(Node(
        package='usb_cam', executable='usb_cam_node_exe', output='screen',
        name=node_name,
        # namespace=ns,
        parameters=[params_path],
        remappings=[
            ("__ns", "/picam360")
        ]
    ))
    return ld

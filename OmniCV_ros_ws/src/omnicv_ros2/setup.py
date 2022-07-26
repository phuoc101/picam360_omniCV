import os
from glob import glob
from setuptools import setup

package_name = 'omnicv_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='phuoc10',
    maintainer_email='nguyenthuanphuoc101@protonmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'show_img = omnicv_ros2.show_img:main',
            'fisheye2rect = omnicv_ros2.show_img_rect_cupy:main',
            '360view = omnicv_ros2.show_img_360view:main',
            'fisheye2rect_ori = omnicv_ros2.show_img_rect_ori:main',
        ],
    },
)

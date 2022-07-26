# picam360_omniCV

ROS2 (Foxy) package base on [OmniCV-Lib](https://github.com/kaustubh-sadekar/OmniCV-Lib) to work with the [Picam360 Panoramic camera](https://www.picam360.com/).\
Added acceleration with CuPy and OpenCV+Cuda to speed up processing.

## Dependencies:
- ROS2 (Foxy)
- CuPy
- OpenCV (build from source with CUDA support)

## Installation
Clone the repository:
```
git clone https://github.com/phuoc101/picam360_omniCV
```
Build the package:
```
cv picam360_omniCV/OmniCV_ros_ws/src/omnicv_ros2 \n
colcon build --package-select omnicv_ros2
```
To test the package:
```
# fisheye to equirectangular images (with GPU acceleration)
ros2 launch omnicv_ros2 ros2 launch omnicv_ros2 picam360_fisheye2rect.launch.py
# 360-degree viewer
ros2 launch omnicv_ros2 picam360_360view.launch.py
# fisheye to equirectangular images (CPU)
ros2 launch omnicv_ros2 ros2 launch omnicv_ros2 picam360_fisheye2rect_ori.launch.py
```

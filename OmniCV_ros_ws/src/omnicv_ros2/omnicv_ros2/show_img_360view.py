import cv2
import numpy as np
import rclpy
import time
import sys
import os

from rclpy.node import Node
from rclpy.logging import LoggingSeverity
from ament_index_python import get_package_share_directory, get_package_prefix
from sensor_msgs.msg import Image
ROOT = os.path.dirname(os.path.abspath(__file__))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from omnicv_cupy import fisheyeImgConvGPU


class PicamReader(Node):
    def __init__(self):
        super().__init__('picam_projection')
        self.get_logger().set_level(LoggingSeverity.DEBUG)
        self.param_file_path = os.path.join(get_package_share_directory('omnicv_ros2'), "config/fisheyeParams.txt")
        self.frame = None
        self.cam_subs = self.create_subscription(
            Image,
            '/picam360/image_raw',
            self.image_callback,
            100)
        self.WINDOW_NAME = "360 Viewer"

        cv2.namedWindow(self.WINDOW_NAME)

        cv2.createTrackbar("FOV", self.WINDOW_NAME, 90, 150, self.nothing)
        cv2.createTrackbar("theta", self.WINDOW_NAME, 180, 360, self.nothing)
        cv2.createTrackbar("phi", self.WINDOW_NAME, 180, 360, self.nothing)

        self.declare_parameter('image_height', 480)
        self.declare_parameter('image_width', 640)
        self.image_size = (self.get_parameter('image_height').value, self.get_parameter('image_width').value)
        self.declare_parameter('out_width', 800)
        self.declare_parameter('out_height', 400)
        self.out_shape = (self.get_parameter('out_height').value, self.get_parameter('out_width').value)

    def nothing(self, x):
        pass

    def image_callback(self, msg):
        if msg.step * msg.height != len(msg.data):
            self.get_logger().debug("bad step/height/data size")
            return

        if msg.encoding == "rgb8":
            dirty = (self.frame is None or msg.width != self.frame.shape[1] or
                     msg.height != self.frame.shape[0] or len(self.frame.shape) < 2 or
                     self.frame.shape[2] != 3)
            if dirty:
                self.frame = np.zeros([msg.height, msg.width, 3], dtype=np.uint8)
            self.frame[:, :, 2] = np.array(msg.data[0::3]).reshape(self.image_size)
            self.frame[:, :, 1] = np.array(msg.data[1::3]).reshape(self.image_size)
            self.frame[:, :, 0] = np.array(msg.data[2::3]).reshape(self.image_size)
        elif msg.encoding == "mono8":
            self.frame = np.array(msg.data).reshape(self.image_size)
        else:
            self.get_logger().debug("unsupported encoding {}".format(msg.encoding))
            return
        if self.frame is not None:
            mapper = fisheyeImgConvGPU(self.param_file_path)
            FOV = cv2.getTrackbarPos("FOV", self.WINDOW_NAME)
            theta = cv2.getTrackbarPos("theta", self.WINDOW_NAME) - 180
            phi = cv2.getTrackbarPos("phi", self.WINDOW_NAME) - 180
            start = time.perf_counter()

            self.frame = mapper.fisheye2equirect(self.frame, self.out_shape)
            self.frame = mapper.equirect2persp(
                self.frame, FOV, theta, phi, 400, 400
            )  # Specify parameters(FOV, theta, phi, height, width)
            self.frame = cv2.flip(self.frame, 1)

            cv2.imshow(self.WINDOW_NAME, self.frame)
            end = time.perf_counter()
            elapsed = end - start
            self.get_logger().info(f"Processing time: {elapsed*1000} ms")
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    picam_projection = PicamReader()

    try:
        rclpy.spin(picam_projection)
    except KeyboardInterrupt:
        pass

    picam_projection.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

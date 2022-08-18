import cv2
import numpy as np
import rclpy
import time
import os
import sys

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
        self.declare_parameter('image_height', 480)
        self.declare_parameter('image_width', 640)
        self.image_size = (self.get_parameter('image_height').value, self.get_parameter('image_width').value)
        self.declare_parameter('out_width', 800)
        self.declare_parameter('out_height', 400)
        self.out_shape = (self.get_parameter('out_height').value, self.get_parameter('out_width').value)

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
            start = time.perf_counter()
            self.frame = mapper.fisheye2equirect(self.frame, self.out_shape)
            self.persp1 = cv2.flip(self.frame, 1)
            cv2.imshow("picam360 with cuda 1", self.persp1)
            self.persp2 = mapper.equirect2cubemap(self.persp1, modif=True)
            self.persp2 = mapper.cubemap2equirect(self.persp2, self.out_shape)
            self.persp2 = self.persp2[0:self.out_shape[0]//2, :]
            cv2.imshow("picam360 with cuda 2", self.persp2)
            end = time.perf_counter()
            elapsed = end - start
            self.get_logger().info(f"Processing time: {elapsed*1000} ms")
            cv2.waitKey(1)
        else:
            self.get_logger().error("No frames received")


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

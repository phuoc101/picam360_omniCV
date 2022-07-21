import cv2
import numpy as np
import rclpy
import time
import os

from rclpy.node import Node
from rclpy.logging import LoggingSeverity
from sensor_msgs.msg import Image
from omnicv import fisheyeImgConv


class PicamReader(Node):
    def __init__(self):
        super().__init__('picam_projection_orig')
        self.get_logger().set_level(LoggingSeverity.DEBUG)
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.param_file_path = os.path.join(self.root, "../config/fisheyeParams.txt")
        self.frame = None
        self.cam_subs = self.create_subscription(
            Image,
            '/picam360/image_raw',
            self.image_callback,
            100)

    def image_callback(self, msg):
        sz = (msg.height, msg.width)
        if msg.step * msg.height != len(msg.data):
            self.get_logger().debug("bad step/height/data size")
            return

        if msg.encoding == "rgb8":
            dirty = (self.frame is None or msg.width != self.frame.shape[1] or
                     msg.height != self.frame.shape[0] or len(self.frame.shape) < 2 or
                     self.frame.shape[2] != 3)
            if dirty:
                self.frame = np.zeros([msg.height, msg.width, 3], dtype=np.uint8)
            self.frame[:, :, 2] = np.array(msg.data[0::3]).reshape(sz)
            self.frame[:, :, 1] = np.array(msg.data[1::3]).reshape(sz)
            self.frame[:, :, 0] = np.array(msg.data[2::3]).reshape(sz)
        elif msg.encoding == "mono8":
            self.frame = np.array(msg.data).reshape(sz)
        else:
            self.get_logger().debug("unsupported encoding {}".format(msg.encoding))
            return
        if self.frame is not None:
            outShape = [400, 800]
            cv2.imshow("original image", self.frame)
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

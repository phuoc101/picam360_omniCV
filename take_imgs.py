import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import cv2
import numpy as np
import os
import logging


class ExamineImage(Node):
    def __init__(self):
        super().__init__('examine_image')

        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(self.dir, 'imgs')
        logging.debug(self.path)
        self.mat = None
        self.subF = self.create_subscription(
            Image,
            '/picam360/image_raw',
            self.image_callback,
            100)
        self.Ts = 0.01
        self.imCnt = 0

    def image_callback(self, msg):
        sz = (msg.height, msg.width)
        # print(msg.header.stamp)
        # if False:
        #     print("{encoding} {width} {height} {step} {data_size}".format(
        #         encoding=msg.encoding, width=msg.width, height=msg.height,
        #         step=msg.step, data_size=len(msg.data)))
        if msg.step * msg.height != len(msg.data):
            logging.error("bad step/height/data size")
            return

        if msg.encoding == "rgb8":
            dirty = (self.mat is None or msg.width != self.mat.shape[1] or
                     msg.height != self.mat.shape[0] or len(self.mat.shape) < 2 or
                     self.mat.shape[2] != 3)
            if dirty:
                self.mat = np.zeros([msg.height, msg.width, 3], dtype=np.uint8)
            self.mat[:, :, 2] = np.array(msg.data[0::3]).reshape(sz)
            self.mat[:, :, 1] = np.array(msg.data[1::3]).reshape(sz)
            self.mat[:, :, 0] = np.array(msg.data[2::3]).reshape(sz)
        elif msg.encoding == "mono8":
            self.mat = np.array(msg.data).reshape(sz)
        else:
            logging.error("unsupported encoding {}".format(msg.encoding))
            return
        if self.mat is not None:
            cv2.imshow("picam360", self.mat)
            if cv2.waitKey(1) == ord('q'):
                cv2.imwrite(os.path.join(
                    self.path, f'img{self.imCnt}.jpg'), self.mat)
                logging.info(f"img{self.imCnt}.jpg taken")
                logging.debug(self.mat.shape)
                self.imCnt += 1

def main(args=None):
    logging.basicConfig(
        format='[%(levelname)s]: %(message)s', level=logging.DEBUG)
    # frameCnt = 0
    rclpy.init(args=args)

    examine_image = ExamineImage()
    try:
        rclpy.spin(examine_image)
    except KeyboardInterrupt:
        pass
    examine_image.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

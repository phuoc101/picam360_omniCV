import cv2
import numpy as np
import rclpy
import time

from rclpy.node import Node
from sensor_msgs.msg import Image
from omnicv import fisheyeImgConv


class ExamineImage(Node):
    def __init__(self):
        super().__init__('examine_image')

        self.param_file_path = "fisheyeParams.txt"
        self.img = None
        self.subF = self.create_subscription(
            Image,
            '/picam360/image_raw',
            self.image_callback,
            100)

    def image_callback(self, msg):
        sz = (msg.height, msg.width)
        # print(msg.header.stamp)
        # if False:
        #     print("{encoding} {width} {height} {step} {data_size}".format(
        #         encoding=msg.encoding, width=msg.width, height=msg.height,
        #         step=msg.step, data_size=len(msg.data)))
        if msg.step * msg.height != len(msg.data):
            print("bad step/height/data size")
            return

        if msg.encoding == "rgb8":
            dirty = (self.img is None or msg.width != self.img.shape[1] or
                     msg.height != self.img.shape[0] or len(self.img.shape) < 2 or
                     self.img.shape[2] != 3)
            if dirty:
                self.img = np.zeros([msg.height, msg.width, 3], dtype=np.uint8)
            self.img[:, :, 2] = np.array(msg.data[0::3]).reshape(sz)
            self.img[:, :, 1] = np.array(msg.data[1::3]).reshape(sz)
            self.img[:, :, 0] = np.array(msg.data[2::3]).reshape(sz)
        elif msg.encoding == "mono8":
            self.img = np.array(msg.data).reshape(sz)
        else:
            print("unsupported encoding {}".format(msg.encoding))
            return
        if self.img is not None:
            # start = time.time()
            # outShape = [400, 800]
            # mapper = fisheyeImgConv(self.param_file_path)
            # self.img = mapper.fisheye2equirect(self.img, outShape)
            # self.img = mapper.equirect2cubemap(self.img, modif=True)
            # self.img = mapper.cubemap2equirect(self.img, outShape)
            # self.img = mapper.cubemap2persp(self.img, FOV=90, THETA=0, PHI=0, Hd=outShape[0], Wd=outShape[1])
            cv2.imshow("picam360", self.img)
            # print(time.time()- start)
            cv2.waitKey(1)


def main(args=None):
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

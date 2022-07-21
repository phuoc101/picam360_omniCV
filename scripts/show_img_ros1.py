import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import time
import os

from sensor_msgs.msg import Image
from omnicv_cupy import fisheyeImgConvGPU

root = os.path.dirname(os.path.abspath(__file__))
file_path = "../config/fisheyeParams.txt"
param_file_path = os.path.join(root, file_path)
def image_reader():
    rospy.init_node("picam_reader", anonymous=True)
    rospy.loginfo("Starting Image Reader")
    img_sub = rospy.Subscriber("/picam360/image_raw", Image, image_callback)
    rospy.spin()

def image_callback(msg):
    bridge = CvBridge()
    # print(msg.data)
    imgMsg = bridge.imgmsg_to_cv2(msg, 'bgr8')
    outShape = [400, 800]
    # mapper = fisheyeImgConvGPU(param_file_path)
    # t = time.perf_counter()
    # imgMsg = mapper.fisheye2equirect(imgMsg, outShape, aperture=183, dely=-9)
    cv2.imshow("picam360", imgMsg)
    # imgMsg = mapper.equirect2cubemap(imgMsg, modif=True)
    # imgMsg = mapper.cubemap2equirect(imgMsg, outShape)
    # imgMsg = imgMsg[0:outShape[0]//2, :]
    # cv2.imshow("picam360 with cuda 2", imgMsg)
    # print(time.perf_counter()- t)
    cv2.waitKey(1)

def main():
    image_reader()

if __name__ == "__main__":
    main()

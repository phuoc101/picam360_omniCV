#!/usr/bin/env/python
import cv2
import sys
import os
ROOT = os.path.relpath(os.path.join(os.path.dirname(__file__), '../omnicv_ros2'))
if ROOT not in sys.path:
    sys.path.append(ROOT)
from omnicv_cupy import fisheyeImgConvGPU


def nothing(x):
    pass


source_path = sys.argv[1]  # =Path to the input file
video = int(sys.argv[2])  # =True if the file is video file
flag = int(sys.argv[3])  # =0 for equirectangular image,
# =1 for fisheye image, =2 for horizontaly placed fisheye image

WINDOW_NAME = "360 Viewer"
root = os.path.dirname(os.path.abspath(__file__))
file_path = "fisheyeParams.txt"
param_file_path = os.path.join(root, file_path)

if flag == 0:
    mapper = fisheyeImgConvGPU()
else:
    mapper = fisheyeImgConvGPU(param_file_path)

cv2.namedWindow(WINDOW_NAME)

cv2.createTrackbar("FOV", WINDOW_NAME, 90, 150, nothing)
cv2.createTrackbar("theta", WINDOW_NAME, 180, 360, nothing)
cv2.createTrackbar("phi", WINDOW_NAME, 180, 360, nothing)

if video:
    cap = cv2.VideoCapture(source_path)
    ret, img = cap.read()
    cv2.imshow("input image", img)
    cv2.waitKey(1)
else:
    img = cv2.imread(source_path)
    ret = True
    cv2.imshow("input image", img)
    cv2.waitKey(1)

while ret:
    if video:
        ret, img = cap.read()
        cv2.imshow("input image", img)
        cv2.waitKey(1)
    if ret:
        FOV = cv2.getTrackbarPos("FOV", WINDOW_NAME)
        theta = cv2.getTrackbarPos("theta", WINDOW_NAME) - 180
        phi = cv2.getTrackbarPos("phi", WINDOW_NAME) - 180

        if flag == 2:
            frame2 = mapper.fisheye2equirect(img, [400, 800])
            frame3 = mapper.equirect2cubemap(frame2, side=256, modif=True)
            output = mapper.cubemap2persp(
                frame3, FOV, theta, phi, 400, 400
            )  # Specify parameters(FOV, theta, phi, height, width)
        if flag == 1:
            frame2 = mapper.fisheye2equirect(img, outShape=[400, 800])
            output = mapper.equirect2persp(frame2, FOV, theta, phi, 400, 400)
            output = cv2.flip(output, 1)
        if flag == 0:
            output = mapper.equirect2persp(img, FOV, theta, phi, 400, 400)

        cv2.imshow(WINDOW_NAME, output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

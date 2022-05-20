#!/usr/bin/env/python
import cv2
import sys
import os
from omnicv import fisheyeImgConv

Img_path = sys.argv[1]
root = os.path.dirname(os.path.abspath(__file__))
file_path = "../config/fisheyeParams.txt"
param_file_path = os.path.join(root, file_path)

frame = cv2.imread(Img_path)
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 400, 400)
cv2.imshow("image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

outShape = [200, 400]
inShape = frame.shape[:2]
print(inShape)

# Uncomment the block to run a perticular experiment

"""
# In case of fisheye lens placed vertically
start = time.time()
mapper = fisheyeImgConv(param_file_path)
frame2 = mapper.fisheye2equirect(frame,outShape)
conv_time = time.time() - start
start = time.time()

# Use the below line if there are multiple images
# and the mapping is not changing in case of a video

for i in range(10):
    frame2 = mapper.applyMap(0,frame)

remap_time = (time.time() - start)/10.0
total_time = conv_time + remap_time
print("time consumed :",total_time)
print("apply remap",remap_time)
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.resizeWindow("image",400,800)
cv2.imshow("image2",frame2)
cv2.waitKey(0)
"""

# """
# In case of fisheye lens placed horizontally
# Note: This conversion is really helpful when we look at the perspective view.
mapper = fisheyeImgConv(param_file_path)
frame2 = mapper.fisheye2equirect(frame, outShape)
frame2 = mapper.equirect2cubemap(frame2, modif=True)
frame2 = mapper.cubemap2equirect(frame2, outShape)
# Hd, Wd = outShape
# FOV = 120
# Phi = 0
# Theta = 0
# frame2 = mapper.equirect2persp(frame2, FOV, Theta, Phi, Hd, Wd)
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 800, 400)
cv2.imshow("image", frame2)
cv2.waitKey(0)
# """

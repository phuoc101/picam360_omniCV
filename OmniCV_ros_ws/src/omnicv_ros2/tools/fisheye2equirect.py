#!/usr/bin/env python3

import cv2
import sys
import os
import argparse
import logging
from ament_index_python import get_package_share_directory
ROOT = os.path.relpath(os.path.join(os.path.dirname(__file__), '../omnicv_ros2'))  # nopep8
if ROOT not in sys.path:  # nopep8
    sys.path.append(ROOT)  # nopep8
from omnicv_cupy import fisheyeImgConvGPU  # nopep8
DF_PARAM = os.path.join(get_package_share_directory(
    'omnicv_ros2'), "config/fisheyeParams.txt")


def main(opts):
    param_path = opts.param
    with open(param_path) as f:
        radius = f.readline()
        aperture = f.readline()
        delx = f.readline()
        dely = f.readline()
    mapper = fisheyeImgConvGPU(param_path)
    outShape = opts.out_shape
    src = cv2.imread(opts.source)
    cv2.imshow('source', src)
    res = mapper.fisheye2equirect(src, outShape, aperture=aperture, dely=dely)
    res = mapper.equirect2cubemap(res, modif=True)
    res = mapper.cubemap2equirect(res, outShape)
    res = res[0:outShape[0]//2, :]
    cv2.imshow('res', res)
    while cv2.waitKey(1) != ord('q'):
        pass
    cv2.imwrite(opts.output, res)


if __name__ == "__main__":
    logging.basicConfig(
        format='[%(levelname)s]: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '--src', '-s', type=str,
                        default=None, help='Image to be rectified')
    parser.add_argument('--output', '--out', '-o', type=str,
                        default=None, help='path to camera parameter file')
    parser.add_argument('--param', type=str, default=str(DF_PARAM),
                        help='path to camera parameter file')
    parser.add_argument('--out_shape', nargs=2, type=int, default=[800, 1600],
                        help='output image size')
    opts = parser.parse_args()
    if not os.path.isfile(opts.source):
        logging.error("Please enter valid source file")
    if opts.output == None:
        logging.error("Enter valid output name")
    main(opts)

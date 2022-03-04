import cv2
import torch
import open3d as o3d
import matplotlib.pyplot as plt

#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-07-19

import sys
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from utils_aug import (
    val_calcu,
    trans3D,
)

# panorama_d1 = np.array(
#     cv2.imread(
#         "/home/li/Documents/LI/SR/srgan-pytorch/test/compare/imHR.png",
#         cv2.IMREAD_GRAYSCALE,
#     )
# )
# panorama_d1 = np.load("imHR.npy")
# panorama_d = np.load("/media/li/ssd1/dense-mpo/Data/Depth/class5_set01_scan001.npy")
# panorama_d1 = np.load("/home/li/Documents/LI/SR/srgan-pytorch/test/compare/imSR.npy")
# panorama_d3 = np.load(
#     "/home/li/Documents/LI/SR/srgan-pytorch/test/compare/class4/imHR.npy"
# )
# panorama_s = np.load("/home/li/Documents/LI/SR/srgan-pytorch/Trans/imHR.npy")
# panorama_r = np.load(sys.argv[2])


W = 1285
H = 439


def crop(array, size):
    crop_W, crop_H = size
    sh = (H - crop_H) // 2
    sw = (W - crop_W) // 2
    eh = sh + crop_H
    ew = sw + crop_W
    return array[sh:eh, sw:ew]


def calcu(size):
    crop_W, crop_H = size
    FARO_RANGE_V_Ori = 123.5
    FARO_RANGE_H_Ori = 360.0
    if size == [384, 384]:
        CropH_rad = (H - crop_H) / 2 / H * FARO_RANGE_V_Ori
        CropW_rad = (W - crop_W) / 2 / W * FARO_RANGE_H_Ori

        FARO_RANGE_V_Crop = 123.5 - 2 * CropH_rad
        FARO_RANGE_H_Crop = 360.0 - 2 * CropW_rad
        return (FARO_RANGE_V_Crop, FARO_RANGE_H_Crop, CropH_rad, CropW_rad)
    else:
        return (FARO_RANGE_V_Ori, FARO_RANGE_H_Ori, 180, 0)


def transt3D(x, size):
    FARO_V, FARO_H, CropH_rad, CropW_rad = size
    panorama_d = x
    h, w = panorama_d.shape
    points = []
    for i, j in product(range(h), range(w)):
        # if panorama_d[i, j] != 0:
        points.append((i, j, panorama_d[i, j]))
    points = np.asarray(points)

    yaw = np.radians(FARO_H * points[:, 1] / w + CropW_rad)
    pitch = np.radians(FARO_V * points[:, 0] / h + CropH_rad)
    value = points[:, 2]

    X = value * np.sin(yaw) * np.sin(pitch)
    Y = value * np.cos(yaw) * np.sin(pitch)
    Z = value * np.cos(pitch)
    XYZ = np.vstack((X, Y, Z))
    XYZ = XYZ.transpose(1, 0)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(XYZ)
    # print(pcd)
    # o3d.visualization.draw_geometries([pcd])
    return XYZ


if __name__ == "__main__":
    panorama_d1 = torch.from_numpy(
        np.load("/home/li/Documents/LI/SR/srgan-pytorch/imHR.npy")
    )
    panorama_d2 = torch.from_numpy(
        np.load("/home/li/Documents/LI/SR/srgan-pytorch/imSR.npy")
    )
    size = [int(sys.argv[1]), int(sys.argv[1])]

    # panorama_d1 =
    #     cv2.resize(panorama_d1, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

    # if size == [384, 384]:
    #     panorama_d1 = crop(panorama_d1, size)
    calcu_size = calcu(size)
    # print(calcu_size)
    XYZ_HR = transt3D(panorama_d1, calcu_size)
    XYZ_SR = transt3D(panorama_d2, calcu_size)
    print(np.max(XYZ_HR - XYZ_SR, 0))
    # vrange = val_calcu(size)
    # XYZ = trans3D(panorama_d1, vrange)
    # print(wrong - right)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(XYZ)
    # print(pcd)
    # o3d.visualization.draw_geometries([pcd])

import torch
import math
import time
from torch import nn
from torchvision.models.vgg import vgg16
import numpy as np
from itertools import product
import torch.nn.functional as F
from torch.autograd import Variable
from pdb import set_trace as brk
import cv2

from libs.modules.chamfer_distance.chamfer_distance import ChamferDistance


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def deg2rad(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError(
                "Input type is not a torch.Tensor. Got {}".format(type(tensor))
            )

        return tensor * math.pi / 180.0

    def cal_offset(self, sh, sw):
        cropH, crop_W = [384, 384]
        W = 1285
        H = 438
        FARO_RANGE_V_Ori = 123.5
        # FARO_VOFFSET_Ori = -33.5
        FARO_RANGE_H_Ori = 360.0

        CropW_rad = sw / W * FARO_RANGE_H_Ori
        CropH_rad = sh / H * FARO_RANGE_V_Ori
        FARO_RANGE_V_Crop = FARO_RANGE_V_Ori * cropH / H
        # FARO_VOFFSET_Crop = -33.5 + CropH_rad
        FARO_RANGE_H_Crop = FARO_RANGE_H_Ori * crop_W / W

        return FARO_RANGE_V_Crop, FARO_RANGE_H_Crop, CropW_rad, CropH_rad

    def transform(self, x, sh, sw):
        x = np.squeeze(x, axis=1)

        b, h, w = x.shape
        x = x.view(b, -1)
        points = []
        points = torch.FloatTensor(points)
        FARO_RANGE_V_Crop, FARO_RANGE_H_Crop, CropW_rad, CropH_rad = self.cal_offset(
            sh, sw
        )

        for z in range(b):
            p, q = torch.meshgrid(torch.arange(h), torch.arange(w))
            points_hw = torch.stack([p, q], dim=-1)
            points_hw = points_hw.view(-1, 2).float()

            # breakpoint(

            points_hw = torch.FloatTensor(points_hw)
            yaw_deg = -FARO_RANGE_H_Crop * points_hw[:, 1] / w + CropW_rad

            pitch_deg = -FARO_RANGE_V_Crop * points_hw[:, 0] / h + CropH_rad
            yaw = self.deg2rad(yaw_deg)
            pitch = self.deg2rad(pitch_deg)

            X = x[z] * torch.sin(yaw) * torch.sin(pitch)
            Y = x[z] * torch.cos(yaw) * torch.sin(pitch)
            Z = x[z] * torch.cos(pitch)
            XYZ = torch.stack((X, Y, Z))

            XYZ = XYZ.transpose(0, 1)
            XYZ = XYZ[None, ...]
        points = torch.cat((points, XYZ), 0)
        # breakpoint()
        points = torch.FloatTensor(points)

        return points

    def __call__(self, fake, tar, sh, sw):
        chamfer_dist = ChamferDistance()
        points = self.transform(tar, sh, sw).cuda()
        points_reconstructed = self.transform(fake, sh, sw).cuda()

        dist1, dist2 = chamfer_dist(points, points_reconstructed)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))

        return loss


if __name__ == "__main__":

    chamfer_dist = ChamferDistance()
    imSR = np.load("test/compare/imSR.npy")
    imSR = torch.FloatTensor(imSR)
    imHR = np.array(cv2.imread("test/compare/imHR.png", cv2.IMREAD_GRAYSCALE))
    imHR = torch.FloatTensor(imHR)
    imSR = imSR[None, None, ...]
    imHR = imHR[None, None, ...]
    loss = ChamferLoss()
    loss = loss(imSR, imHR, 27, 450)

    print(loss)

# version 2.21


# VERSION 1
# class ChamferLoss(nn.Module):
#     def __init__(self):
#         super(ChamferLoss, self).__init__()
#         self.use_cuda = torch.cuda.is_available()

#     def cal_offset(self, sh, sw):
#         cropH, crop_W = [384, 384]
#         W = 1285
#         H = 438
#         FARO_RANGE_V_Ori = 123.5
#         # FARO_VOFFSET_Ori = -33.5
#         FARO_RANGE_H_Ori = 360.0

#         CropW_rad = sw / W * FARO_RANGE_H_Ori
#         CropH_rad = sh / H * FARO_RANGE_V_Ori
#         FARO_RANGE_V_Crop = FARO_RANGE_V_Ori * cropH / H
#         # FARO_VOFFSET_Crop = -33.5 + CropH_rad
#         FARO_RANGE_H_Crop = FARO_RANGE_H_Ori * crop_W / W

#         return FARO_RANGE_V_Crop, FARO_RANGE_H_Crop, CropW_rad, CropH_rad

#     def transform(self, x, sh, sw):
#         x = np.squeeze(x, axis=1)
#         b, h, w = x.shape

#         points = []

#         FARO_RANGE_V_Crop, FARO_RANGE_H_Crop, CropW_rad, CropH_rad = self.cal_offset(
#             sh, sw
#         )

#         for z in range(b):
#             points_hw = []
#             for i, j in product(range(h), range(w)):
#                 points_hw.append((i, j, x[z, i, j]))
#             points_hw = np.asarray(points_hw)
#             yaw = np.radians(-FARO_RANGE_H_Crop * points_hw[:, 1] / w + CropW_rad)
#             pitch = np.radians(-FARO_RANGE_V_Crop * points_hw[:, 0] / h + CropH_rad)
#             drange = points_hw[:, 2]

#             X = drange * np.sin(yaw) * np.sin(pitch)
#             Y = drange * np.cos(yaw) * np.sin(pitch)
#             Z = drange * np.cos(pitch)
#             XYZ = np.vstack((X, Y, Z))
#             XYZ = XYZ.transpose()
#             points.append((XYZ))
#         points = np.asarray(points)
#         points = torch.from_numpy(points).float()

#         return points

#     def batch_pairwise_dist(self, x, y):
#         _, num_points_x, _ = x.shape  # bs, num_points, points_dim
#         _, num_points_y, _ = y.shape
#         xx = torch.bmm(x, x.transpose(2, 1))
#         yy = torch.bmm(y, y.transpose(2, 1))
#         zz = torch.bmm(x, y.transpose(2, 1))
#         if self.use_cuda:
#             dtype = torch.cuda.LongTensor
#         else:
#             dtype = torch.LongTensor
#         diag_ind_x = torch.arange(0, num_points_x).type(dtype)
#         diag_ind_y = torch.arange(0, num_points_y).type(dtype)
#         # brk()
#         rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
#         ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
#         P = rx.transpose(2, 1) + ry - 2 * zz
#         return P

#     def __call__(self, fake, tar, sh, sw):
#         gts = self.transform(tar, sh, sw)
#         preds = self.transform(fake, sh, sw)
#         P = self.batch_pairwise_dist(gts, preds)
#         mins, _ = torch.min(P, 1)
#         loss_1 = torch.sum(mins)
#         mins, _ = torch.min(P, 2)
#         loss_2 = torch.sum(mins)

#         return loss_1 + loss_2


# if __name__ == "__main__":
#     im = np.load("im.npy")
#     # img = np.array(cv2.imread("img2.png",cv2.IMREAD_GRAYSCALE))
#     im = im[None, None, ...]
#     # img = img[None,None,...]

#     loss = ChamferLoss()
#     loss = loss(im, im, 27, 450)
#     print(loss)


# VERSION2
# class ChamferLoss(nn.Module):
# def __init__(self):
#     super(ChamferLoss, self).__init__()
#     self.use_cuda = torch.cuda.is_available()

# def deg2rad(self, tensor):
#     if not torch.is_tensor(tensor):
#         raise TypeError(
#             "Input type is not a torch.Tensor. Got {}".format(type(tensor))
#         )

#     return tensor * math.pi / 180.0

# def cal_offset(self, sh, sw):
#     cropH, crop_W = [384, 384]
#     W = 1285
#     H = 438
#     FARO_RANGE_V_Ori = 123.5
#     # FARO_VOFFSET_Ori = -33.5
#     FARO_RANGE_H_Ori = 360.0

#     CropW_rad = sw[0] / W * FARO_RANGE_H_Ori
#     CropH_rad = sh[0] / H * FARO_RANGE_V_Ori
#     FARO_RANGE_V_Crop = FARO_RANGE_V_Ori * cropH / H
#     # FARO_VOFFSET_Crop = -33.5 + CropH_rad
#     FARO_RANGE_H_Crop = FARO_RANGE_H_Ori * crop_W / W

#     return FARO_RANGE_V_Crop, FARO_RANGE_H_Crop, CropW_rad, CropH_rad

# def transform(self, x, sh, sw):
#     x = np.squeeze(x, axis=1)
#     b, h, w = x.shape

#     points = []
#     points = torch.FloatTensor(points)
#     FARO_RANGE_V_Crop, FARO_RANGE_H_Crop, CropW_rad, CropH_rad = self.cal_offset(
#         sh, sw
#     )

#     for z in range(b):
#         points_hw = []
#         for i, j in product(range(h), range(w)):
#             points_hw.append((i, j, x[z, i, j]))
#         points_hw = torch.FloatTensor(points_hw)
#         # breakpoint()
#         yaw_deg = -FARO_RANGE_H_Crop * points_hw[:, 1] / w + CropW_rad
#         pitch_deg = -FARO_RANGE_V_Crop * points_hw[:, 0] / h + CropH_rad
#         yaw = self.deg2rad(yaw_deg)
#         pitch = self.deg2rad(pitch_deg)
#         drange = points_hw[:, 2]

#         X = drange * torch.sin(yaw) * torch.sin(pitch)
#         Y = drange * torch.cos(yaw) * torch.sin(pitch)
#         Z = drange * torch.cos(pitch)
#         XYZ = torch.stack((X, Y, Z))

#         XYZ = XYZ.transpose(0, 1)
#         XYZ = XYZ[None, ...]
#         points = torch.cat((points, XYZ), 0)
#     # breakpoint()
#     points = torch.FloatTensor(points)

#     return points

# def forward(self, fake, tar, sh, sw):
#     chamfer_dist = ChamferDistance()
#     points = self.transform(tar, sh, sw).cuda()
#     points_reconstructed = self.transform(fake, sh, sw).cuda()
#     dist1, dist2 = chamfer_dist(points, points_reconstructed)
#     loss = (torch.mean(dist1)) + (torch.mean(dist2))
#     return loss


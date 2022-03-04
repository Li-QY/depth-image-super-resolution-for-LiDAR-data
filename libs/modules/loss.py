import logging
import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16
from itertools import product
from torch.autograd import Variable
from pdb import set_trace as brk

from libs.modules.chamfer_distance.chamfer_distance import ChamferDistance

log = logging.getLogger(__name__)


class AdversarialLoss(nn.Module):
    def __init__(self, metric="mse"):
        super().__init__()
        self.register_buffer("label", torch.tensor(1.0))
        if metric == "mse":
            self.loss = nn.MSELoss()
        elif metric == "bce":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError()
        print("Adversarial loss:", self.loss.__class__.__name__)

    def __call__(self, input, target):
        loss = 0
        if isinstance(input, list):
            for i in input:
                target = (self.label * target).expand_as(i)
                loss += self.loss(i, target)
            loss /= len(input)
        else:
            target = (self.label * target).expand_as(input)
            loss += self.loss(input, target)
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self, metric="l1"):
        super().__init__()
        vgg = vgg16(pretrained=True)
        vgg = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        if metric == "l1":
            self.loss = nn.L1Loss()
        elif metric == "l2":
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError()
        self.register_buffer(
            "mean", torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        print("Perceptual loss:", self.loss.__class__.__name__)

    def preprocess(self, imgs):
        imgs = (imgs + 1.0) / 2.0
        # 1ch -> 3ch
        imgs = imgs - self.mean
        imgs = imgs / self.std
        return imgs

    def forward(self, imgs, targets):
        imgs = self.vgg(self.preprocess(imgs))
        targets = self.vgg(self.preprocess(targets))
        return self.loss(imgs, targets)


class PixelLoss(nn.Module):
    def __init__(self, metric="l1"):
        super().__init__()
        if metric == "l1":
            self.loss = nn.L1Loss(reduction="none")
        elif metric == "l2":
            self.loss = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError()
        print("Pixel loss:", self.loss.__class__.__name__)

    def forward(self, imgs, targets, masks=None):
        loss = self.loss(imgs, targets).flatten(1)
        if masks is not None:
            masks = masks.float().flatten(1)
            loss = loss * masks
            loss = loss.sum(dim=1) / masks.sum(dim=1)
        return loss.mean()


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.chamfer_dist = ChamferDistance()

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
        # x = np.squeeze(x, axis=1)

        b, _, h, w = x.shape
        x = x.reshape(b, -1)
        points = []
        FARO_RANGE_V_Crop, FARO_RANGE_H_Crop, CropW_rad, CropH_rad = self.cal_offset(
            sh, sw
        )

        p, q = torch.meshgrid(torch.arange(h), torch.arange(w))
        points_hw = torch.stack([p, q], dim=-1)
        points_hw = points_hw.view(-1, 2).float().cuda()

        yaw_deg = -FARO_RANGE_H_Crop * points_hw[:, 1] / w + CropW_rad
        pitch_deg = -FARO_RANGE_V_Crop * points_hw[:, 0] / h + CropH_rad
        yaw = self.deg2rad(yaw_deg)
        pitch = self.deg2rad(pitch_deg)

        sin_yaw = torch.sin(yaw)
        cos_yaw = torch.cos(yaw)
        sin_pitch = torch.sin(pitch)
        cos_pitch = torch.cos(pitch)

        for z in range(b):

            X = x[z] * sin_yaw * sin_pitch
            Y = x[z] * cos_yaw * sin_pitch
            Z = x[z] * cos_pitch
            XYZ = torch.stack((X, Y, Z))

            XYZ = XYZ.transpose(0, 1)
            XYZ = XYZ[None, ...]
            points.append(XYZ)

        points = torch.cat(points)
        return points

    def forward(self, fake, tar, crop_start):
        if len(fake.shape) == 3:
            dist1, dist2 = self.chamfer_dist(tar, fake)
            loss = (torch.mean(dist1)) + (torch.mean(dist2))
            return loss
        else:
            sh, sw = crop_start
            points = self.transform(tar, sh, sw).cuda()
            points_reconstructed = self.transform(fake, sh, sw).cuda()
            dist1, dist2 = self.chamfer_dist(points, points_reconstructed)
            loss = (torch.mean(dist1)) + (torch.mean(dist2))
            return loss


class CartesianSmoothnessLoss(nn.Module):
    def __init__(self, norm=True):
        super().__init__()
        self.norm = norm
        log.info(f"Gradient Smoothness")

    def gradient_h(self, tensor):
        tensor = F.pad(tensor, (0, 0, 0, 1), mode="replicate")
        gradient = tensor[:, :, :-1, :] - tensor[:, :, 1:, :]
        return gradient

    def gradient_w(self, tensor):
        tensor = F.pad(tensor, (0, 1, 0, 0), mode="replicate")
        gradient = tensor[:, :, :, :-1] - tensor[:, :, :, 1:]
        return gradient

    def forward(self, ortho_xyz, img=None):

        # if self.norm:
        #     mean = ortho_xyz.mean(dim=(2, 3), keepdim=True)
        #     ortho_xyz = ortho_xyz / (mean + 1e-8)

        dh_xyz = torch.abs(self.gradient_h(ortho_xyz))
        dw_xyz = torch.abs(self.gradient_w(ortho_xyz))
        dh = dh_xyz.sum(dim=1, keepdim=True)
        dw = dw_xyz.sum(dim=1, keepdim=True)

        if img is not None:
            dh_img = torch.abs(self.gradient_h(img))
            dw_img = torch.abs(self.gradient_w(img))
            dh_I = torch.mean(dh_img, dim=1, keepdim=True)
            dw_I = torch.mean(dw_img, dim=1, keepdim=True)
            weight_h = torch.exp(-dh_I)
            weight_w = torch.exp(-dw_I)

            dh *= weight_h
            dw *= weight_w

        return dh.mean() + dw.mean()


class EMDloss:
    def __init__(self):
        super(EMDloss, self).__init__()
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

        CropW_rad = sw[0] / W * FARO_RANGE_H_Ori
        CropH_rad = sh[0] / H * FARO_RANGE_V_Ori
        FARO_RANGE_V_Crop = FARO_RANGE_V_Ori * cropH / H
        # FARO_VOFFSET_Crop = -33.5 + CropH_rad
        FARO_RANGE_H_Crop = FARO_RANGE_H_Ori * crop_W / W

        return FARO_RANGE_V_Crop, FARO_RANGE_H_Crop, CropW_rad, CropH_rad

    def transform(self, x, sh, sw):
        # x = np.squeeze(x, axis=1)

        b, _, h, w = x.shape
        x = x.view(b, -1)
        points = []
        FARO_RANGE_V_Crop, FARO_RANGE_H_Crop, CropW_rad, CropH_rad = self.cal_offset(
            sh, sw
        )

        p, q = torch.meshgrid(torch.arange(h), torch.arange(w))
        points_hw = torch.stack([p, q], dim=-1)
        points_hw = points_hw.view(-1, 2).float().cuda()

        yaw_deg = -FARO_RANGE_H_Crop * points_hw[:, 1] / w + CropW_rad
        pitch_deg = -FARO_RANGE_V_Crop * points_hw[:, 0] / h + CropH_rad
        yaw = self.deg2rad(yaw_deg)
        pitch = self.deg2rad(pitch_deg)

        sin_yaw = torch.sin(yaw)
        cos_yaw = torch.cos(yaw)
        sin_pitch = torch.sin(pitch)
        cos_pitch = torch.cos(pitch)

        for z in range(b):

            X = x[z] * sin_yaw * sin_pitch
            Y = x[z] * cos_yaw * sin_pitch
            Z = x[z] * cos_pitch
            XYZ = torch.stack((X, Y, Z))

            XYZ = XYZ.transpose(0, 1)
            XYZ = XYZ[None, ...]
            points.append(XYZ)

        points = torch.cat(points)

        return points

    def forward(self, fake, tar, sh, sw):
        points = self.transform(tar, sh, sw).cuda()
        points_reconstructed = self.transform(fake, sh, sw).cuda()
        loss = earth_mover_distance(points, points_reconstructed)
        return loss

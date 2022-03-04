#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-07-19
# import seaborn as sns


class transform:
    def __init__(self, x):
        cropH = 384
        crop_W = 384
        W = 1285
        H = 438
        FARO_RANGE_V_Ori = 123.5
        # FARO_VOFFSET_Ori = -33.5
        FARO_RANGE_H_Ori = 360.0
        CropH_rad = (H - cropH) / 2 / H * FARO_RANGE_V_Ori
        CropW_rad = (W - crop_W) / 2 / W * FARO_RANGE_H_Ori
        FARO_RANGE_V_Crop = 123.5 - 2 * CropH_rad
        # FARO_VOFFSET_Crop = -33.5 + CropH_rad
        FARO_RANGE_H_Crop = 360.0 - 2 * CropW_rad
        self.b, _, h, w = x.shape
        x = x.view(self.b, -1)

        p, q = torch.meshgrid(torch.arange(h), torch.arange(w))
        points_hw = torch.stack([p, q], dim=-1)
        points_hw = points_hw.view(-1, 2).float().cuda()

        yaw_deg = -FARO_RANGE_H_Crop * points_hw[:, 1] / w + CropW_rad
        pitch_deg = -FARO_RANGE_V_Crop * points_hw[:, 0] / h + CropH_rad
        yaw = self.deg2rad(yaw_deg)
        pitch = self.deg2rad(pitch_deg)

        self.sin_yaw = torch.sin(yaw)
        self.cos_yaw = torch.cos(yaw)
        self.sin_pitch = torch.sin(pitch)
        self.cos_pitch = torch.cos(pitch)

    def deg2rad(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError(
                "Input type is not a torch.Tensor. Got {}".format(type(tensor))
            )

        return tensor * math.pi / 180.0

    def __call__(self, x):
        points = []
        X = x[0] * self.sin_yaw * self.sin_pitch
        Y = x[0] * self.cos_yaw * self.sin_pitch
        Z = x[0] * self.cos_pitch
        XYZ = torch.stack((X, Y, Z))

        XYZ = XYZ.transpose(0, 1)
        return points


if __name__ == "__main__":
    pass


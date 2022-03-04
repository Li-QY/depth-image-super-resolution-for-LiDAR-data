# https://github.com/leftthomas/SRGAN/blob/master/model.py
import math
import time

import torch
from torch import nn
from torch.nn import init
from libs.modules.partialconv2d import PartialConv2d

# Added


def init_weights(init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "uniform":
                init.uniform_(m.weight.data)
            elif init_type == "eye":
                init.eye_(m.weight.data)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    return init_func


def conv3x3(in_planes, out_planes, kernel_size, padding, stride):
    """3x3 convolution with padding"""
    return PartialConv2d(
        in_planes,
        out_planes,
        kernel_size,
        padding=padding,
        stride=stride,
        bias=True,
        return_mask=True,
        multi_channel=False,
    )


class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))
        super(Generator, self).__init__()

        self.block1 = conv3x3(in_ch, 64, kernel_size=9, padding=4, stride=1)
        self.block1_1 = nn.PReLU()
        block2 = self._make_layer(ResidualBlock, 64, 5)
        self.block2 = nn.Sequential(*block2)

        self.block3 = conv3x3(64, 64, kernel_size=3, padding=1, stride=1)
        self.block3_1 = nn.BatchNorm2d(64)

        block4_1 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        self.block4_1 = nn.Sequential(*block4_1)
        self.block4_2 = conv3x3(64, out_ch, kernel_size=9, padding=4, stride=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(planes))
        for i in range(1, blocks):
            layers.append(block(planes))
        return layers

    def forward(self, x, mask):
        block1, mask = self.block1(x, mask)
        block1 = self.block1_1(block1)
        block2 = block1
        for m in self.block2:
            block2, mask = m(block2, mask)
        block3, mask = self.block3(block2, mask)
        block3 = self.block3_1(block3)
        block4 = block1 + block3
        for m in self.block4_1:
            block4, mask = m(block4, mask)
        block4, mask = self.block4_2(block4, mask)
        mask = mask.bool()
        mask = ~mask
        block5 = torch.masked_fill(torch.tanh(block4), mask, -1)
        return block5


class Discriminator(nn.Module):
    def __init__(self, in_ch):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(channels, channels,
                             kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(channels, channels,
                             kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x, mask):
        residual, mask = self.conv1(x, mask)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual, mask = self.conv2(residual, mask)
        residual = self.bn2(residual)
        return x + residual, mask


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = conv3x3(
            in_channels,
            in_channels * up_scale ** 2,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.mask_up = nn.Upsample(scale_factor=up_scale, mode="nearest")
        self.prelu = nn.PReLU()

    def forward(self, x, mask):
        x, mask = self.conv(x, mask)
        x = self.pixel_shuffle(x)
        mask = self.mask_up(mask)
        x = self.prelu(x)
        return x, mask


# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.prelu = nn.PReLU()
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(channels)

#     def forward(self, x):
#         residual = self.conv1(x)
#         residual = self.bn1(residual)
#         residual = self.prelu(residual)
#         residual = self.conv2(residual)
#         residual = self.bn2(residual)
#         return x + residual


# class UpsampleBLock(nn.Module):
#     def __init__(self, in_channels, up_scale):
#         super(UpsampleBLock, self).__init__()
#         self.conv = nn.Conv2d(
#             in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1
#         )
#         self.pixel_shuffle = nn.PixelShuffle(up_scale)
#         self.prelu = nn.PReLU()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.pixel_shuffle(x)
#         x = self.prelu(x)
#         return x

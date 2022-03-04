import collections
import os
import os.path as osp
import pickle as pkl
import time
from glob import glob
import torch.nn.functional as F

# import cv2

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data
from torchvision import transforms as transforms
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset
from tqdm import tqdm

Sequence = collections.abc.Sequence
_VALID_DEPTH_MIN = 0.3
_VALID_DEPTH_MAX = 120.0

H = 1757 // 4
W = 5140 // 4

FARO_RANGE_H = 360.0
FARO_RANGE_V = 123.5
FARO_OFFSET_V = -33.5

_MODAL = {
    # name, [dir, extension, range]
    "rgb": ["RGB", ".png", [0.0, 255.0]],
    "depth": ["Depth", ".npy", [0.0, 120.0]],
    "disp": [None, None, [0.0, 1.0 / _VALID_DEPTH_MIN]],
    "normal": ["Normal", ".npy", [-1.0, 1.0]],
    "refl": ["Reflectance", ".npy", [0.0, 1.0]],
    "mask": [None, None, [0.0, 1.0]],
}


def depth_to_ortho(depth):
    """
    Polar-to-orthogonal coordinate conversion
    """

    _, _, H, W = depth.shape
    device = depth.device

    polar_h = torch.arange(H, device=device) / float(H - 1)
    polar_h = polar_h.view(1, 1, H, 1).expand(1, 1, H, W)
    polar_h = (1.0 - polar_h) * FARO_RANGE_V + FARO_OFFSET_V
    polar_h = polar_h * np.pi / 180.0

    polar_w = torch.arange(W, device=device) / float(W - 1)
    polar_w = polar_w.view(1, 1, 1, W).expand(1, 1, H, W)
    polar_w = polar_w * FARO_RANGE_H
    polar_w = polar_w * np.pi / 180.0

    ortho_x = depth * torch.cos(polar_h) * torch.cos(polar_w)
    ortho_y = depth * torch.cos(polar_h) * torch.sin(polar_w)
    ortho_z = depth * torch.sin(polar_h)

    # polar_img = torch.cat((polar_w, polar_h, depth), dim=1)
    ortho_coord = torch.cat((ortho_x, ortho_y, ortho_z), dim=1)

    return ortho_coord


def invert_depth(depth):
    """
    depth: original range
    invert_depth: original range
    """
    valid = depth > _VALID_DEPTH_MIN
    valid *= depth < _VALID_DEPTH_MAX
    depth[valid] = 1.0 / depth[valid]
    depth[~valid] = 0.0
    return depth


def revert_depth(inverce_depth):
    """
    invert_depth: [0, 1]
    depth: [0, 1]
    """
    inverce_depth *= 1 / _VALID_DEPTH_MIN
    valid = inverce_depth > 1 / _VALID_DEPTH_MAX
    inverce_depth[valid] = 1 / inverce_depth[valid]
    inverce_depth[~valid] = 0.0
    depth = inverce_depth / _VALID_DEPTH_MAX
    return depth


class DenseMPO(VisionDataset):
    def __init__(
        self,
        root,
        setlist,
        transforms=None,
        transform=None,
        target_transform=None,
        required=None,
    ):
        super().__init__(
            root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.data_root = osp.join(root, "Data")
        with open(setlist, "r") as lines:
            # 生成按行的去掉空格的元素列表
            self.scans = list(map(lambda x: x.rstrip(), lines))
        assert required is None or isinstance(required, Sequence)
        if required is None:
            self.required = ["rgb", "depth", "normal", "refl"]
        else:
            assert all([r in ["rgb", "depth", "normal", "refl"] for r in required])
            self.required = required

    def _set_path(self, modal, scan_id):
        return osp.join(self.data_root, _MODAL[modal][0], scan_id + _MODAL[modal][1])

    # def _img(self, img_path):
    #     return cv2.imread(img_path, cv2.IMREAD_COLOR)[..., ::-1]

    def _npy(self, npy_path):
        return np.load(npy_path)

    def _load(self, modal, scan_id):
        path = self._set_path(modal, scan_id)
        if _MODAL[modal][1] == ".npy":
            return self._npy(path)
        # elif _MODAL[modal][1] == ".png":
        #     return self._img(path)

    def __len__(self):
        return len(self.scans)

    def extra_repr(self):
        return "Required: {}".format(self.required)

    def __getitem__(self, index):
        scan_id = self.scans[index]
        scan_dict = {}
        for modal in self.required:
            scan_dict[modal] = self._load(modal, scan_id)
        if "depth" in self.required:

            # clamp the depth values in [0.1, 120]
            valid = scan_dict["depth"] > _VALID_DEPTH_MIN
            valid *= scan_dict["depth"] < _VALID_DEPTH_MAX
            scan_dict["depth"] *= valid
            # Invert the clamped depth
            disparity = scan_dict["depth"].copy()
            disparity[valid] = 1.0 / disparity[valid]
            scan_dict["disp"] = disparity

            scan_dict["mask"] = np.int32(valid)

        return scan_dict


class Augmentation:
    def __init__(self, device, transform):
        self.device = device
        self.transform = transform
        assert self.transform.vrange is None or len(self.transform.vrange) == 2
        self.vmin, self.vmax = [0, 1.0] if not transform.vrange else transform.vrange  #
        if not isinstance(self.transform.crop_size, Sequence):  # 将传入的数值改为序列
            crop_size = (transform.crop_size, transform.crop_size)
        self.crop_H, self.crop_W = crop_size
        assert self.transform.crop_mode in ["center", "random"]
        self.crop_mode = self.transform.crop_mode
        self.H, self.W = transform.base_size
        self.interp_factor = transform.interp_factor
        self.interp_mode = transform.interp_mode
        format_string = (
            self.__class__.__name__
            + "(base_size=({}, {}), crop_size=({}, {}), crop_mode={}, vrange=({}, {}))".format(
                self.H,
                self.W,
                self.crop_H,
                self.crop_W,
                self.crop_mode,
                self.vmin,
                self.vmax,
            )
        )
        print(format_string)

    def _reformation(self, array):
        if len(array.shape) == 4:
            array = array.permute(0, 3, 1, 2)
        elif len(array.shape) == 3:
            array = array[:, None, ...]
        return array.float()

    def _crop(self, array, sh, sw):
        eh = sh + self.crop_H
        ew = sw + self.crop_W
        return array[:, :, sh:eh, sw:ew]  # 图片的向量

    def _normalize(self, array, modal):
        vmin, vmax = _MODAL[modal][2]
        array = (array - vmin) / (vmax - vmin)  # [0, 1]
        array = array * (self.vmax - self.vmin) + self.vmin
        return array

    def __call__(self, data, modal, flip, crop_start):
        data = data.to(self.device)
        sh, sw = crop_start  # crop_start
        if flip != 0.0:
            if modal == "mask":
                data = self._reformation(data)
                # data = self._crop(data, sh, sw)
                data_HR = torch.flip(data, [3])
            else:
                data = self._reformation(self._normalize(data, modal))
                # data = self._crop(data, sh, sw)
                data_HR = torch.flip(data, [3])
        else:
            if modal == "mask":
                data = self._reformation(data)
                # data_HR = self._crop(data, sh, sw)
            else:
                data = self._reformation(self._normalize(data, modal))
                # data_HR = self._crop(data, sh, sw)
        data_LR = F.interpolate(
            data_HR, scale_factor=1.0 / self.interp_factor, mode=self.interp_mode
        )
        # , recompute_scale_factor=True
        return data_HR, data_LR


class midPointSet(Dataset):
    def __init__(self, label, SRblocks, targets):
        self.block = {}
        self.block["SR"] = SRblocks
        self.block["HR"] = targets
        self.block["label"] = label

    def __len__(self):
        return len(self.block["SR"])

    def __getitem__(self, index):
        scene = {}
        scene["SR"] = self.block["SR"][index]
        scene["HR"] = self.block["HR"][index]
        scene["label"] = self.block["label"][index]
        return scene


if __name__ == "__main__":
    SR = torch.rand(30, 3, 5)
    HR = torch.rand(30, 3, 5)
    _, label = torch.meshgrid(torch.arange(3), torch.arange(10))
    label = label.reshape(-1)
    newset = midPointSet(label, SR, HR)
    train_loader = torch.utils.data.DataLoader(
        newset, batch_size=3, shuffle=True, num_workers=2, drop_last=True,
    )
    for scene in train_loader:
        print(scene.shape)

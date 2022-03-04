import collections
import os
import os.path as osp
import pickle as pkl
import random
import time
from glob import glob

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data
from torchvision import transforms as transforms
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm

Sequence = collections.abc.Sequence

_MODAL = {
    # name, [dir, extension, range]
    "rgb": ["RGB", ".png", [0.0, 255.0]],
    "depth": ["Depth", ".npy", [0.0, 120.0]],
    "normal": ["Normal", ".npy", [-1.0, 1.0]],
    "refl": ["Reflectance", ".npy", [0.0, 1.0]],
    "mask": [None, None, [0.0, 1.0]],
}


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
            self.scans = list(map(lambda x: x.rstrip(), lines))  # 生成按行的去掉空格的元素列表
        assert required is None or isinstance(required, Sequence)
        if required is None:
            self.required = ["rgb", "depth", "normal", "refl"]
        else:
            assert all([r in ["rgb", "depth", "normal", "refl"] for r in required])
            self.required = required

    def _set_path(self, modal, scan_id):
        return osp.join(self.data_root, _MODAL[modal][0], scan_id + _MODAL[modal][1])

    def _img(self, img_path):
        return cv2.imread(img_path, cv2.IMREAD_COLOR)[..., ::-1]

    def _npy(self, npy_path):
        return np.load(npy_path)

    def _load(self, modal, scan_id):
        path = self._set_path(modal, scan_id)
        if _MODAL[modal][1] == ".npy":
            return self._npy(path)
        elif _MODAL[modal][1] == ".png":
            return self._img(path)

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
            scan_dict["mask"] = np.int32(scan_dict["depth"] != 0)
            # scan_dict["mask"] = np.zeros((439, 1285))
        if self.transform:
            scan_dict = self.transform(scan_dict)  # crop_list
        return scan_dict


class JointPreprocess(object):
    def __init__(self, base_size, crop_size, flip, crop_mode=None, vrange=None):
        assert vrange is None or len(vrange) == 2
        self.vmin, self.vmax = [0, 1.0] if not vrange else vrange  #
        if not isinstance(crop_size, Sequence):  # 将传入的数值改为序列
            crop_size = (crop_size, crop_size)
        self.crop_H, self.crop_W = crop_size
        assert isinstance(base_size, Sequence)
        self.H, self.W = base_size
        assert self.H >= self.crop_H, "{}, {}".format(self.H, self.crop_H)  # 输出
        assert self.W >= self.crop_W, "{}, {}".format(self.W, self.crop_W)
        assert crop_mode in ["center", "random"]
        self.crop_mode = crop_mode
        self.resize_kwargs = {
            "dsize": (self.W, self.H),
            "interpolation": cv2.INTER_NEAREST,
        }
        self.flip = flip

    def _sample_topleft(self):  # 算单侧剪裁大小 a开始剪裁b大小
        if self.crop_mode == "center":
            sh = (self.H - self.crop_H) // 2
            sw = (self.W - self.crop_W) // 2
        else:
            sh = random.randint(0, self.H - self.crop_H)
            sw = random.randint(0, self.W - self.crop_W)
        return sh, sw

    def _crop(self, array, sh, sw):
        eh = sh + self.crop_H
        ew = sw + self.crop_W
        return array[sh:eh, sw:ew]  # 图片的向量

    def _normalize(self, array, modal):
        vmin, vmax = _MODAL[modal][2]
        array = (array - vmin) / (vmax - vmin)  # [0, 1]
        array = array * (self.vmax - self.vmin) + self.vmin
        return array

    def _to_tensor(self, array):
        if len(array.shape) == 3:
            array = array.transpose(2, 0, 1)
        elif len(array.shape) == 2:
            array = array[None, ...]
        return torch.from_numpy(array).float()

    def __call__(self, scan_dict):
        sh, sw = self._sample_topleft()
        crop_start = [sh, sw]  # crop_start
        for modal, scan in scan_dict.items():
            if self.flip:
                if modal == "mask":
                    scan = self._crop(cv2.resize(scan, **self.resize_kwargs), sh, sw)
                    scan = self._to_tensor(scan)
                    scan_dict[modal] = torch.flip(scan, [0])
                else:
                    scan = self._crop(cv2.resize(scan, **self.resize_kwargs), sh, sw)
                    scan = self._to_tensor(self._normalize(scan, modal))
                    scan_dict[modal] = torch.flip(scan, [0])
            else:
                if modal == "mask":
                    scan = self._crop(cv2.resize(scan, **self.resize_kwargs), sh, sw)
                    scan_dict[modal] = self._to_tensor(scan)
                else:
                    scan = self._crop(cv2.resize(scan, **self.resize_kwargs), sh, sw)
                    scan = self._to_tensor(self._normalize(scan, modal))
                    scan_dict[modal] = scan

        # print(scan_dict["depth"][0])
        scan_dict["list"] = crop_start
        return scan_dict

    def __repr__(self):
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
        return format_string


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    root = "/home/li/Documents/LI/dense-mpo"
    setlist = "/home/li/Documents/LI/dense-mpo/Data/all.txt"
    crop_list = []
    H = 1757 // 4
    W = 5140 // 4
    base_size = (H, W)
    transform = JointPreprocess(
        base_size=base_size,
        crop_size=384,
        crop_mode="center",
        vrange=[-1.0, 1.0],
        flip=None,
    )
    dataset = DenseMPO(root, setlist, crop_list, transform=transform)
    print(dataset)
    loader = data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True
    )

    for imgs in tqdm(loader):
        result = []
        for modal, img in imgs.items():
            if len(img.shape) == 3:
                img = img[:, None].float()
            print(img.min(), img.max())
            img -= img.min()
            img /= img.max()
            result.append(
                torchvision.utils.make_grid(img).numpy().transpose(1, 2, 0)
            )  # ??????
        result = np.vstack(result)
        plt.imshow(result)
        plt.show()
        quit()

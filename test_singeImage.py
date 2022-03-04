import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as utils

from libs.modules.DF_mask import Generator, init_weights

gen_init = "/home/li/Documents/LI/SR/srgan-pytorch/models/DF_mask/G_epoch_02440.pth"
testD_path = "/home/li/Documents/LI/dense-mpo/Data/Depth/class0_set01_scan011.npy"
testR_path = "/home/li/Documents/LI/dense-mpo/Data/Reflectance/class0_set01_scan011.npy"

crop_size1 = (384, 384)
crop_size2 = (96, 96)
crop_H, crop_W = crop_size1
VMIN, VMAX = [-1.0, 1.0]


def scale(imgs):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (imgs + 1.0) / 2.0


def _crop(array, sh, sw):
    eh = sh + crop_H
    ew = sw + crop_W
    return array[sh:eh, sw:ew]  # 图片的向量


def _normalize(array):
    vmin, vmax = [0.0, 120.0]
    array = (array - vmin) / (vmax - vmin)  # [0, 1]
    array = array * (VMAX - VMIN) + VMIN
    return array


def _to_tensor(array):
    if len(array.shape) == 3:
        array = array.transpose(2, 0, 1)
        array = array[None, ...]
    elif len(array.shape) == 2:
        array = array[None, None, ...]
    return torch.from_numpy(array).float()


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # Interpolation
    interp_factor = 4
    interp_mode = "nearest"
    downsample = lambda x: F.interpolate(
        x, scale_factor=1.0 / interp_factor, mode=interp_mode
    )

    im = np.load(testD_path)
    imR = np.load(testR_path)

    H = 1757 // 4
    W = 5140 // 4
    dsize = (W, H)
    im = cv2.resize(im, dsize=dsize, interpolation=cv2.INTER_NEAREST)
    imR = cv2.resize(imR, dsize=dsize, interpolation=cv2.INTER_NEAREST)

    # crop and resize
    sh = (H - crop_H) // 2
    sw = (W - crop_W) // 2

    im = _crop(im, sh, sw)
    imR = _crop(imR, sh, sw)

    cv2.imwrite("imHR.png", im)
    im = _normalize(im)
    imR = _normalize(imR)

    imHR = scale(im) * 120
    imHRR = scale(imR) * 120

    np.save("imHR.npy", imHR)

    im = _to_tensor(im)
    im = downsample(im)
    imR = _to_tensor(imR)
    imR = downsample(imR)

    mask_LR = torch.from_numpy(np.int32(np.array(im) != 0)).float()
    # Generate
    utils.save_image(im, "imLR.png")
    n_ch = 1
    G = Generator(n_ch, n_ch, interp_factor)
    if gen_init is not None:
        print("Init:", gen_init)
        state_dict = torch.load(gen_init)
        G.load_state_dict(state_dict)
    # G.cuda()

    imgs_SR = G(scale(im), scale(imR), mask_LR)
    imgs_SR = imgs_SR[0, 0, :]

    imgs_SR = scale(imgs_SR)
    print(imgs_SR.max(), imgs_SR.min())
    utils.save_image(imgs_SR, "imgSR.png")
    im = im[0, 0, :]

    imgs_SR = imgs_SR.cpu().numpy()

    im = np.array(scale(im) * 120)

    np.save("imLR.npy", im)
    imgs_SR = imgs_SR * 120
    np.save("imSR.npy", imgs_SR)
    # torch.clamp(, 0.0, 1.0)
    # cv2.resize(im, dsize=crop_size2, fx=1/4, fy=1/4, interpolation=cv2.INTER_NEAREST)

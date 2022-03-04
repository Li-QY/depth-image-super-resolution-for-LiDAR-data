import math
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import yaml
from addict import Dict
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from libs.modules.loss import AdversarialLoss, PerceptualLoss, PixelLoss, TVLoss
from libs.modules.srgan import Discriminator, Generator, init_weights
from libs.modules.ssim import SSIM
from utils import evaluator, get_device, prepare_dataloader, set_requires_grad


def scale(imgs):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (imgs + 1.0) / 2.0


if __name__ == "__main__":
    # Load a yaml configuration file
    config = Dict(yaml.load(open(sys.argv[1]), Loader=yaml.SafeLoader))
    device = get_device(config.cuda)

    # Dataset
    modal = config.modal
    train_loader, val_loader = prepare_dataloader(config)

    # Interpolation
    interp_factor = config.train.interp_factor
    interp_mode = config.train.interp_mode
    downsample = lambda x: F.interpolate(
        x, scale_factor=1.0 / interp_factor, mode=interp_mode
    )

    # Model setup
    n_ch = config.train.n_ch
    G = Generator(n_ch, n_ch, interp_factor)
    G.to(device)

    # Loss for training
    criterion_pix = PixelLoss("l2").to(device)

    # Optimizer
    optim_G = optim.Adam(G.parameters(), lr=config.train.lr, betas=(0.9, 0.999))

    # Experiemtn ID
    exp_id = config.experiment_id

    # Tensorboard
    writer = SummaryWriter("runs/" + exp_id)
    os.makedirs(osp.join("models", exp_id), exist_ok=True)
    print("Experiment ID:", exp_id)

    # Train the generator
    n_epoch = math.ceil(config.train.n_iter / len(train_loader))
    for epoch in tqdm(range(1, n_epoch + 1), desc="Epoch"):
        # Training
        G.train()
        for iteration, imgs_HR in tqdm(
            enumerate(train_loader, 1),
            desc="Training/Iteration",
            total=len(train_loader),
            leave=False,
        ):
            imgs_HR = imgs_HR[modal].to(device)
            imgs_LR = downsample(imgs_HR)

            optim_G.zero_grad()

            # Generate fake images
            imgs_SR = G(scale(imgs_LR))
            loss_pix = criterion_pix(imgs_SR, imgs_HR)
            loss_pix.backward()
            optim_G.step()

            step = (epoch - 1) * len(train_loader) + iteration
            writer.add_scalar("Loss/Generator/Image", loss_pix.item(), step)
            for i, o in enumerate(optim_G.param_groups):
                writer.add_scalar("LR/Generator/group_{}".format(i), o["lr"], step)

        # Validation
        mse, ssim, psnr, summary = evaluator(val_loader, G, device, config, step)
        writer.add_images("Results", summary, step)
        writer.add_scalar("Score/MSE", mse, step)
        writer.add_scalar("Score/SSIM", ssim, step)
        writer.add_scalar("Score/PSNR", psnr, step)

        if epoch % config.train.freq_save == 0:
            torch.save(
                G.state_dict(),
                osp.join("models", exp_id, "G_epoch_{:05d}.pth".format(epoch)),
            )

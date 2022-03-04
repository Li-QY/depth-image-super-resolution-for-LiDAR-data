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
from tensorboard.plugins.mesh import summary as mesh_summary
from libs.modules.loss import AdversarialLoss, PixelLoss

# ,PerceptualLoss , TVLoss, ChamferLoss
from libs.modules.srgan_mask_skip import Discriminator, Generator, init_weights
from libs.modules.ssim import SSIM
from utils import (
    evaluator,
    get_device,
    prepare_dataloader,
    set_requires_grad,
    val_calcu,
)


def scale(imgs):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (imgs + 1.0) / 2.0


def umiscale(imgs):
    return imgs * 2.0 - 1.0


if __name__ == "__main__":
    # Load a yaml configuration file
    config = Dict(yaml.load(open(sys.argv[1]), Loader=yaml.SafeLoader))
    device = get_device(config.cuda)

    # Dataset
    modal = config.modal[0]
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

    def hook(name):
        def _hook(module, input, output):
            isnan = torch.any(torch.isnan(output[0]))
            print(name, isnan)

        return _hook

    for name, module in G.named_modules():
        module.register_forward_hook(hook(name))
    if config.train.gen_init is not None:
        print("Init:", config.train.gen_init)
        state_dict = torch.load(config.train.gen_init)
        G.load_state_dict(state_dict)
    # G.apply(init_weights)
    G.to(device)
    D = Discriminator(n_ch)
    if config.train.dis_init is not None:
        print("Init:", config.train.dis_init)
        state_dict = torch.load(config.train.dis_init)
        D.load_state_dict(state_dict)
    D.to(device)

    # Loss for training
    criterion_adv = AdversarialLoss("bce").to(device)
    # criterion_vgg = PerceptualLoss("l2").to(device)
    criterion_pix = PixelLoss("l1").to(device)
    # criterion_var = TVLoss().to(device)
    # criterion_cham = ChamferLoss().to(device)

    # precalcu
    # vrange = val_calcu(384)
    # colors_tensor = torch.zeros([384, 384])
    # colors_tensor = colors_tensor.view(-1)
    # colors_tensor = colors_tensor[None, ...]
    # colors_tensor = torch.cat((colors_tensor, colors_tensor, colors_tensor), dim=1)
    # colors_tensor = colors_tensor[None, ...]

    # Optimizer
    optim_G = optim.Adam(G.parameters(), lr=config.train.lr, betas=(0.9, 0.999))
    optim_D = optim.Adam(D.parameters(), lr=config.train.lr, betas=(0.9, 0.999))
    scheduler_G = MultiStepLR(optim_G, config.train.lr_steps, config.train.lr_decay)
    scheduler_D = MultiStepLR(optim_D, config.train.lr_steps, config.train.lr_decay)

    # Experiemtn ID
    exp_id = config.experiment_id

    # Print necessary information
    print("Mask: In")
    print("srgan_mask_skip")

    # Tensorboard
    writer = SummaryWriter("runs/" + exp_id)
    os.makedirs(osp.join("models", exp_id), exist_ok=True)
    print("Experiment ID:", exp_id)

    # Adversarial training
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
            sh, sw = imgs_HR["list"]
            mask = imgs_HR["mask"]
            imgs_HR = imgs_HR[modal].to(device)
            mask_HR = mask.to(device)
            imgs_LR = downsample(imgs_HR)
            mask_LR = downsample(mask_HR)

            # Generate fake images
            imgs_SR = G(scale(imgs_LR), mask_LR)
            # Update the discriminator, mask_LR
            set_requires_grad(D, True)
            optim_D.zero_grad()

            pred_fake = D(imgs_SR.detach())
            pred_real = D(imgs_HR)
            loss_D = criterion_adv(pred_fake, 0.0)
            loss_D += criterion_adv(pred_real, 1.0)
            loss_D /= 2.0
            loss_D.backward()
            optim_D.step()

            # Update the generator
            set_requires_grad(D, False)
            optim_G.zero_grad()

            pred_fake = D(imgs_SR)
            loss_adv = criterion_adv(pred_fake, 1.0)
            loss_pix = criterion_pix(imgs_SR, imgs_HR)
            # loss_cham = criterion_cham(imgs_SR, imgs_HR, sh, sw)
            # loss_cham = criterion_cham(imgs_SR, imgs_HR, sh, sw)
            # loss_vgg = criterion_vgg(imgs_SR, imgs_HR)
            # loss_var = criterion_var(imgs_SR)
            loss_G = loss_pix
            loss_G += 1e-3 * loss_adv
            # loss_G += 6e-3 * loss_vgg
            # loss_G += 2e-8 * loss_var
            # loss_G += 1e2 * loss_cham
            loss_G.backward()
            optim_G.step()

            step = (epoch - 1) * len(train_loader) + iteration
            writer.add_scalar("Loss/Discriminator/Adversarial", loss_D.item(), step)
            writer.add_scalar("Loss/Generator/Adversarial", loss_adv.item(), step)
            writer.add_scalar("Loss/Generator/Image", loss_pix.item(), step)
            # writer.add_scalar("Loss/Generator/cham", loss_cham.item(), step)
            # writer.add_scalar("Loss/Generator/Perceptual", loss_vgg.item(), step)
            # writer.add_scalar("Loss/Generator/TV", loss_var.item(), step)
            writer.add_scalar("Loss/Generator/Total", loss_G.item(), step)

            for i, o in enumerate(optim_G.param_groups):
                writer.add_scalar("LR/Generator/group_{}".format(i), o["lr"], step)
            for i, o in enumerate(optim_D.param_groups):
                writer.add_scalar("LR/Discriminatorgroup_{}".format(i), o["lr"], step)

            scheduler_G.step()
            scheduler_D.step()

        # # Validation
        # mse, ssim, psnr, summary = evaluator(val_loader, G, device, config, step)
        # # writer.add_mesh(               , vrange             , vertices_tensor
        # #     "Point_clouds",
        # #     vertices=vertices_tensor,
        # #     colors=colors_tensor,
        # #     global_step=step,
        # # )
        # writer.add_images("Results", summary, step)
        # writer.add_scalar("Score/MSE", mse, step)
        # writer.add_scalar("Score/SSIM", ssim, step)
        # writer.add_scalar("Score/PSNR", psnr, step)

        if epoch % config.train.freq_save == 0:
            torch.save(
                G.state_dict(),
                osp.join("models", exp_id, "G_epoch_{:05d}.pth".format(epoch)),
            )
            torch.save(
                D.state_dict(),
                osp.join("models", exp_id, "D_epoch_{:05d}.pth".format(epoch)),
            )
            torch.save(
                optim_D.state_dict(),
                osp.join("optimizer", exp_id, "D_epoch_{:05d}.pth".format(epoch)),
            )
            torch.save(
                optim_G.state_dict(),
                osp.join("optimizer", exp_id, "G_epoch_{:05d}.pth".format(epoch)),
            )

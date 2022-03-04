import math
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision
import yaml
from addict import Dict
from tensorboard.plugins.mesh import summary as mesh_summary
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from apex import amp
from libs.modules.DF_mask import Discriminator, Generator, init_weights
from pointnet.pointnet import PointNetSeg
from libs.modules.loss import AdversarialLoss, PixelLoss, ChamferLoss
from libs.modules.ssim import SSIM

from libs.datasets.mpo_crop import Augmentation, midPointSet
from utils_aug import (
    evaluator_DRPN,
    get_device,
    prepare_dataloader,
    set_requires_grad,
    _sample_topleft,
    val_calcu,
    makeTrainBlock,
    scene2blocks,
)

# ,PerceptualLoss , TVLoss,
blankimage = 384 * 384 * (-1)


def scale(imgs):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (imgs + 1.0) / 2.0


def umiscale(imgs):
    return imgs * 2.0 - 1.0


if __name__ == "__main__":
    # Set benchmark
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    torch.backends.cudnn.benchmark = True

    # Load a yaml configuration file
    config = Dict(yaml.load(open(sys.argv[1]), Loader=yaml.SafeLoader))
    device = get_device(config.cuda)

    # Dataset
    modal = config.modal
    train_loader, val_loader = prepare_dataloader(config)
    augmentation_T = Augmentation(device, config.train)
    augmentation_V = Augmentation(device, config.val)

    # Model setup
    n_ch = config.train.n_ch
    G = Generator(n_ch, n_ch, config.train.interp_factor)
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

    PG = PointNetSeg()
    if config.pointnet.gen_init is not None:
        print("Init:", config.pointnet.gen_init)
        state_dict = torch.load(config.pointnet.gen_init)
        PG.load_state_dict(state_dict)
    PG.apply(init_weights)
    PG.to(device)

    # def load(self):

    # Loss for training
    criterion_adv = AdversarialLoss("bce").to(device)
    criterion_pix = PixelLoss("l1").to(device)
    criterion_cham = ChamferLoss().to(device)

    # precalcu
    crop_val = _sample_topleft(config.val)
    vrange = val_calcu(crop_val)

    # Optimizer
    optim_G = optim.Adam(
        list(G.parameters()) + list(PG.parameters()),
        lr=config.train.lr,
        betas=(0.9, 0.999),
    )
    optim_D = optim.Adam(D.parameters(), lr=config.train.lr, betas=(0.9, 0.999))

    if config.train.opG_init is not None:
        print("Init:", config.train.opG_init)
        state_dict = torch.load(config.train.opG_init)
        optim_G.load_state_dict(state_dict)
    if config.train.opD_init is not None:
        print("Init:", config.train.opD_init)
        state_dict = torch.load(config.train.opD_init)
        optim_D.load_state_dict(state_dict)

    # Apex
    [G, D], [optim_G, optim_D] = amp.initialize(
        [G, D,], [optim_G, optim_D], opt_level="O0", num_losses=2, verbosity=0,
    )

    scheduler_G = MultiStepLR(optim_G, config.train.lr_steps, config.train.lr_decay)
    scheduler_D = MultiStepLR(optim_D, config.train.lr_steps, config.train.lr_decay)

    # Experiemtn ID
    exp_id = config.experiment_id

    # Print necessary information
    print("Loss: Pixel*1, Adver*1e-3", "Chamfer*1e2", "Mask: In", "DF_PN")

    # Tensorboard
    writer = SummaryWriter("runs/" + exp_id)
    os.makedirs(osp.join("models", exp_id), exist_ok=True)
    os.makedirs(osp.join("optimizer", exp_id), exist_ok=True)
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

            crop_start = _sample_topleft(config.train)
            mask_HR = imgs_HR["mask"]
            imgs_HR_dep = imgs_HR["depth"]
            imgs_HR_ref = imgs_HR["refl"]
            imgs_HR_dep, imgs_LR_dep = augmentation_T(
                imgs_HR_dep, "depth", config.train.flip, crop_start
            )
            imgs_HR_ref, imgs_LR_ref = augmentation_T(
                imgs_HR_ref, "refl", config.train.flip, crop_start
            )
            mask_HR, mask_LR = augmentation_T(
                mask_HR, "mask", config.train.flip, crop_start
            )

            # Generate fake images
            imgs_SR = G(scale(imgs_LR_dep), scale(imgs_LR_ref), mask_LR)

            # Update the discriminator, mask_LR
            set_requires_grad(D, True)
            optim_D.zero_grad()

            pred_fake = D(imgs_SR.detach())
            pred_real = D(imgs_HR_dep)
            loss_D = criterion_adv(pred_fake, 0.0)
            loss_D += criterion_adv(pred_real, 1.0)
            loss_D /= 2.0
            with amp.scale_loss(loss_D, optim_D, loss_id=0) as loss_D_scaled:
                loss_D_scaled.backward()
            optim_D.step()
            step = (epoch - 1) * len(train_loader) + iteration

            # PointNet

            SRlabels, SRblocks, targets = makeTrainBlock(
                imgs_SR, imgs_HR_dep, crop_start
            )
            pointset = midPointSet(SRlabels, SRblocks, targets)
            point_loader = torch.utils.data.DataLoader(  # Batch_of_PNx3x4096
                pointset,
                batch_size=config.pointnet.batch_size,
                shuffle=True,
                # num_workers=0,
                drop_last=True,
            )

            for P_iteration, points in tqdm(
                enumerate(point_loader, 1),
                desc="PointNet/Iteration",
                total=len(point_loader),
                leave=False,
            ):
                # for points in point_loader:
                SRblocks = points["SR"]
                HRblocks = points["HR"]

                PointsSR = PG(SRblocks)

                loss_cham = criterion_cham(PointsSR, HRblocks, crop_start) / len(
                    point_loader
                )
                with amp.scale_loss(loss_cham, optim_G, loss_id=1) as loss_PG_scaled:
                    loss_PG_scaled.backward(retain_graph=True)
                writer.add_scalar("Loss/Generator/cham", loss_cham.item(), step)

            for i, o in enumerate(optim_G.param_groups):
                writer.add_scalar("LR/Generator/group_{}".format(i), o["lr"], step)
            # Update the generator
            set_requires_grad(D, False)

            pred_fake = D(imgs_SR)
            loss_adv = criterion_adv(pred_fake, 1.0)
            loss_pix = criterion_pix(imgs_SR, imgs_HR_dep)
            loss_G = loss_pix
            loss_G += 1e-3 * loss_adv
            with amp.scale_loss(loss_G, optim_G, loss_id=1) as loss_G_scaled:
                loss_G_scaled.backward()

            # summary = [imgs_HR_dep, imgs_SR]
            # summary = torch.cat(summary, dim=2)[:8]
            # summary = torch.clamp(scale(summary), 0.0, 1.0)

            # writer.add_images("Results/Train", summary, step)
            writer.add_scalar("Loss/Discriminator/Adversarial", loss_D.item(), step)
            writer.add_scalar("Loss/Generator/Adversarial", loss_adv.item(), step)
            writer.add_scalar("Loss/Generator/Pixel", loss_pix.item(), step)
            writer.add_scalar("Loss/Generator/Total", loss_G.item(), step)

            for i, o in enumerate(optim_D.param_groups):
                writer.add_scalar("LR/Discriminatorgroup_{}".format(i), o["lr"], step)
            optim_G.step()
            optim_G.zero_grad()

            scheduler_G.step()
            scheduler_D.step()

        # Validation, vertices_tensor
        (
            mse,
            ssim,
            psnr,
            cham,
            summary,
            SR_tensors,
            HR_tensors,
            colors_tensor,
        ) = evaluator_DRPN(
            val_loader,
            G,
            PG,
            device,
            config,
            step,
            augmentation_V,
            vrange,
            criterion_cham,
        )

        writer.add_images("Results", summary, step)
        writer.add_scalar("Score/MSE", mse, step)
        writer.add_scalar("Score/SSIM", ssim, step)
        writer.add_scalar("Score/PSNR", psnr, step)
        writer.add_scalar("Score/CHAM", cham, step)
        writer.add_mesh(
            "PointClouds/SR",
            vertices=SR_tensors,
            colors=colors_tensor,
            global_step=step,
        )
        writer.add_mesh(
            "PointClouds/HR",
            vertices=HR_tensors,
            colors=colors_tensor,
            global_step=step,
        )

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
                PG.state_dict(),
                osp.join("models", exp_id, "PG_epoch_{:05d}.pth".format(epoch)),
            )
            torch.save(
                optim_D.state_dict(),
                osp.join("optimizer", exp_id, "D_epoch_{:05d}.pth".format(epoch)),
            )
            torch.save(
                optim_G.state_dict(),
                osp.join("optimizer", exp_id, "G_epoch_{:05d}.pth".format(epoch)),
            )

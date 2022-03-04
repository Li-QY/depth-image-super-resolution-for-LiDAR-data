# import open3d as o3d
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# import numpy as np

from tqdm import tqdm
from libs.datasets.mpo_crop import (
    DenseMPO,
    Augmentation,
    midPointSet,
    revert_depth,
)
from libs.modules.ssim import SSIM


def scale(imgs):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (imgs + 1.0) / 2.0


def umiscale(imgs):
    return imgs * 2.0 - 1.0


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return device


def set_requires_grad(net, requires_grad=True):
    for param in net.parameters():
        param.requires_grad = requires_grad


def deg2rad(tensor):
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    return tensor * math.pi / 180.0


def val_calcu(crop):
    sh, sw = crop
    crop_W, crop_H = [384, 384]
    W = 1285
    H = 439
    FARO_RANGE_V_Ori = 123.5
    FARO_RANGE_H_Ori = 360.0
    CropH_rad = sh / H * FARO_RANGE_V_Ori
    CropW_rad = sw / W * FARO_RANGE_H_Ori
    FARO_RANGE_V_range = 123.5 - 2 * CropH_rad
    FARO_RANGE_H_range = 360.0 - 2 * CropW_rad

    p, q = torch.meshgrid(torch.arange(crop_H), torch.arange(crop_W))
    points_hw = torch.stack([p, q], dim=-1)
    points_hw = points_hw.view(-1, 2).float().cuda()

    yaw_deg = FARO_RANGE_H_range * points_hw[:, 1] / crop_W + CropW_rad
    pitch_deg = FARO_RANGE_V_range * points_hw[:, 0] / crop_H + CropH_rad
    yaw = deg2rad(yaw_deg)
    pitch = deg2rad(pitch_deg)

    sin_yaw = torch.sin(yaw)[None, ...]
    cos_yaw = torch.cos(yaw)[None, ...]
    sin_pitch = torch.sin(pitch)[None, ...]
    cos_pitch = torch.cos(pitch)[None, ...]
    vrange = torch.cat([sin_yaw, sin_pitch, cos_yaw, cos_pitch])
    return vrange


def trans3D(x, vrange):
    b, _, _, _ = x.shape
    x = x.reshape(b, -1)
    points = []
    sin_yaw, sin_pitch, cos_yaw, cos_pitch = vrange
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


def _sample_topleft(config):  # 算单侧剪裁大小 a开始剪裁b大小
    H, W = config.base_size
    crop_H, crop_W = (config.crop_size, config.crop_size)
    if config.crop_mode == "center":
        sh = (H - crop_H) // 2
        sw = (W - crop_W) // 2
    else:
        sh = random.randint(0, H - crop_H)
        sw = random.randint(0, W - crop_W)
    return [sh, sw]


def prepare_dataloader(config):
    modal = config.modal
    # base_size = config.train.base_size
    # crop_size = config.train.crop_size
    train_dataset = DenseMPO(
        config.dataset.root, config.dataset.setlist.train, required=modal,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )
    print(train_dataset)

    val_dataset = DenseMPO(
        config.dataset.root, config.dataset.setlist.val, required=modal,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    print(val_dataset)
    return train_loader, val_loader


def test_dataloader(config):
    modal = config.modal

    test_dataset = DenseMPO(
        config.dataset.root, config.dataset.setlist.test, required=modal,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        drop_last=False,
    )
    print(test_dataset)

    return test_loader


def evaluator(val_loader, G, device, config, step, augmentation_V, vrange):

    criterion_mse = nn.MSELoss().to(device)
    criterion_ssim = SSIM().to(device)

    G.eval()
    with torch.no_grad():
        score_mse = 0
        score_ssim = 0
        score_psnr = 0
        n_sample = len(val_loader.dataset)
        for val_iteration, imgs_HR in tqdm(
            enumerate(val_loader, 1),
            desc="Evaluation/Iteration",
            total=len(val_loader),
            leave=False,
        ):
            crop_start = _sample_topleft(config.val)
            mask_HR = imgs_HR["mask"]
            imgs_HR_dep = imgs_HR["depth"]
            flip = random.randint(0, 1)
            imgs_HR_dep, imgs_LR_dep = augmentation_V(
                imgs_HR_dep, "depth", flip, crop_start
            )

            mask_HR, mask_LR = augmentation_V(mask_HR, "mask", flip, crop_start)
            imgs_SR = G(scale(imgs_LR_dep), mask_LR)
            batch_size = imgs_HR_dep.size(0)

            # Compute scores
            # breakpoint()
            mse = criterion_mse(imgs_SR, imgs_HR_dep)
            score_ssim += criterion_ssim(imgs_SR, imgs_HR_dep).item() * batch_size
            score_psnr += 10 * torch.log10(1.0 / mse).item() * batch_size
            score_mse += mse.item() * batch_size
            # Save SR images
            if val_iteration == 1:
                imgs_LR_dep = F.interpolate(
                    imgs_LR_dep, scale_factor=config.train.interp_factor
                )
                summary = [imgs_LR_dep, imgs_HR_dep, imgs_SR]
                summary = torch.cat(summary, dim=2)[:8]
                summary = torch.clamp(scale(summary), 0.0, 1.0)

                ver_SR = trans3D(scale(imgs_SR) * 120, vrange)
                vertices_tensor = ver_SR
            vertices_tensor,

        return (
            score_mse / n_sample,
            score_ssim / n_sample,
            score_psnr / n_sample,
            summary,
            vertices_tensor,
        )


def evaluator_DF(val_loader, G, device, config, step, augmentation_V, vrange):

    criterion_mse = nn.MSELoss().to(device)
    criterion_ssim = SSIM().to(device)

    G.eval()
    with torch.no_grad():
        score_mse = 0
        score_ssim = 0
        score_psnr = 0
        n_sample = len(val_loader.dataset)
        for val_iteration, imgs_HR in tqdm(
            enumerate(val_loader, 1),
            desc="Evaluation/Iteration",
            total=len(val_loader),
            leave=False,
        ):
            crop_start = _sample_topleft(config.val)
            mask_HR = imgs_HR["mask"]
            # disp = imgs_HR["disp"]
            imgs_HR_dep = imgs_HR["depth"]
            imgs_HR_ref = imgs_HR["refl"]
            flip = random.randint(0, 1)
            imgs_HR_dep, imgs_LR_dep = augmentation_V(
                imgs_HR_dep, "depth", flip, crop_start
            )
            imgs_HR_ref, imgs_LR_ref = augmentation_V(
                imgs_HR_ref, "refl", flip, crop_start
            )
            mask_HR, mask_LR = augmentation_V(mask_HR, "mask", flip, crop_start)
            # _, disp_LR = augmentation_V(disp, "disp", config.train.flip, crop_start)
            imgs_SR = G(scale(imgs_LR_dep), scale(imgs_LR_ref), mask_LR)
            # imgs_SR = revert_depth(scale(imgs_SR)) * 2.0 - 1.0
            batch_size = imgs_HR_dep.size(0)

            # Compute scores
            mse = criterion_mse(imgs_SR, imgs_HR_dep)
            score_ssim += criterion_ssim(imgs_SR, imgs_HR_dep).item() * batch_size
            score_psnr += 10 * torch.log10(1.0 / mse).item() * batch_size
            score_mse += mse.item() * batch_size

            # Save SR images
            if val_iteration == 1:
                imgs_LR_dep = F.interpolate(
                    imgs_LR_dep, scale_factor=config.train.interp_factor
                )
                summary = [imgs_HR_ref, imgs_LR_dep, imgs_HR_dep, imgs_SR]
                summary = torch.cat(summary, dim=2)[:8]
                summary = torch.clamp(scale(summary), 0.0, 1.0)

                ver_SR = trans3D(scale(imgs_SR) * 120, vrange)[0]

                colors_tensor = torch.abs(scale(imgs_HR_dep[0]) - scale(imgs_SR[0]))
                colors_tensor = colors_tensor.view(1, 1, 384, 384)
                colors_tensor = colors_tensor.detach().cpu().numpy()

                for array in colors_tensor:
                    array = array.squeeze()  # remove the redundant dim
                    colored_array = cm.jet(array)  # returns as RGBA
                    colored_array = colored_array[..., :3]  # only uses RGB
                colors_tensor = colored_array.reshape(1, -1, 3) * 255.0

                colors_tensor = torch.from_numpy(colors_tensor)

                vertices_tensor = ver_SR[None, ...]

        return (
            score_mse / n_sample,
            score_ssim / n_sample,
            score_psnr / n_sample,
            summary,
            vertices_tensor,
            colors_tensor,
        )


def evaluator_DRPN(
    val_loader, G, PG, device, config, step, augmentation_V, vrange, criterion_cham
):

    criterion_mse = nn.MSELoss().to(device)
    criterion_ssim = SSIM().to(device)

    G.eval()
    with torch.no_grad():
        score_mse = 0
        score_ssim = 0
        score_psnr = 0
        score_cham = 0
        n_sample = len(val_loader.dataset)
        for val_iteration, imgs_HR in tqdm(
            enumerate(val_loader, 1),
            desc="Evaluation/Iteration",
            total=len(val_loader),
            leave=False,
        ):
            crop_start = _sample_topleft(config.val)
            mask_HR = imgs_HR["mask"]
            imgs_HR_dep = imgs_HR["depth"]
            imgs_HR_ref = imgs_HR["refl"]
            imgs_HR_dep, imgs_LR_dep = augmentation_V(
                imgs_HR_dep, "depth", config.train.flip, crop_start
            )
            imgs_HR_ref, imgs_LR_ref = augmentation_V(
                imgs_HR_ref, "refl", config.train.flip, crop_start
            )
            mask_HR, mask_LR = augmentation_V(
                mask_HR, "mask", config.train.flip, crop_start
            )

            imgs_SR = G(scale(imgs_LR_dep), scale(imgs_LR_ref), mask_LR)
            batch_size = imgs_HR_dep.size(0)

            # Compute scores
            # breakpoint()
            mse = criterion_mse(imgs_SR, imgs_HR_dep)
            score_ssim += criterion_ssim(imgs_SR, imgs_HR_dep).item() * batch_size
            score_psnr += 10 * torch.log10(1.0 / mse).item() * batch_size
            score_mse += mse.item() * batch_size

            # PointNet

            SRlabels, SRblocks, targets = makeValBlock(imgs_SR, imgs_HR_dep, crop_start)
            pointset = midPointSet(SRlabels, SRblocks, targets)
            point_loader = torch.utils.data.DataLoader(  # Batch_of_PNx3x4096
                pointset,
                batch_size=config.pointnet.batch_size,
                shuffle=True,
                # num_workers=0,
                drop_last=True,
            )
            for points in point_loader:
                SRblocks = points["SR"]
                HRblocks = points["HR"]
                PointsSR = PG(SRblocks)
                loss_cham = criterion_cham(PointsSR, HRblocks, crop_start) / len(
                    point_loader
                )
            score_cham += loss_cham.item() * batch_size

            # Save SR images
            if val_iteration == 1:
                imgs_LR_dep = F.interpolate(
                    imgs_LR_dep, scale_factor=config.train.interp_factor
                )
                summary = [imgs_HR_ref, imgs_LR_dep, imgs_HR_dep, imgs_SR]
                summary = torch.cat(summary, dim=2)[:8]
                summary = torch.clamp(scale(summary), 0.0, 1.0)

                SR_tensor = PointsSR[0][None, ...]
                HR_tensor = HRblocks[0][None, ...]

                colors_tensor = torch.abs(scale(imgs_HR_dep[0]) - scale(imgs_SR[0]))
                colors_tensor = colors_tensor.view(1, 1, 384, 384)
                colors_tensor = colors_tensor.detach().cpu().numpy()

                for array in colors_tensor:
                    array = array.squeeze()  # remove the redundant dim
                    colored_array = cm.jet(array)  # returns as RGBA
                    colored_array = colored_array[..., :3]  # only uses RGB
                colors_tensor = colored_array.reshape(1, -1, 3) * 255.0

                colors_tensor = torch.from_numpy(colors_tensor)

        return (
            score_mse / n_sample,
            score_ssim / n_sample,
            score_psnr / n_sample,
            score_cham / n_sample,
            summary,
            SR_tensor,
            HR_tensor,
            colors_tensor,
        )


def makeTrainBlock(scene, target, crop_start):
    vrange = val_calcu(crop_start)
    b, _, _, _ = scene.shape
    no0index = (scene != -1.0).view(b, -1).float()  # BxN
    no0index[torch.sum(no0index, 1) == 0.0] = (
        torch.randint(2, (1, 147456)).float().cuda()  # change 0 distribution
    )
    target = trans3D(target, vrange)  # BxNx3
    scene = trans3D(scene, vrange)  # BxNx3
    times = ((torch.sum(no0index, 1) // 4096) // 3 + 1).int()  # B 为了快，减一半
    SRlabels = [torch.ones(times[i]) * i for i in range(b)]  # M
    SRlabels = torch.cat(SRlabels)
    sample_index = [
        torch.multinomial(no0index[i], times[i] * 4096, True) for i in range(b)
    ]  # Bx(times+1)
    SRblocks = [scene[i, sample_index[i], :].view(-1, 4096, 3) for i in range(b)]
    HRblocks = [target[i, sample_index[i], :].view(-1, 4096, 3) for i in range(b)]
    SRblocks = torch.cat(SRblocks).permute(0, 2, 1)  # Mx3*4096
    HRblocks = torch.cat(HRblocks)  # Mx4096*3
    return SRlabels, SRblocks, HRblocks


def makeValBlock(scene, target, crop_start):
    vrange = val_calcu(crop_start)
    b, _, _, _ = scene.shape
    no0index = (scene != -1.0).view(b, -1).float()  # BxN
    target = trans3D(scale(target), vrange)  # BxNx3
    scene = trans3D(scale(scene), vrange)  # BxNx3
    times = (torch.sum(no0index, 1) // 4096 + 1).int()  # B
    SRlabels = [torch.ones(times[i]) * i for i in range(b)]  # M
    SRlabels = torch.cat(SRlabels)
    sample_index = [
        torch.multinomial(no0index[i], times[i] * 4096, True) for i in range(b)
    ]  # Bx(times+1)
    SRblocks = [scene[i, sample_index[i], :].view(-1, 4096, 3) for i in range(b)]
    HRblocks = [target[i, sample_index[i], :].view(-1, 4096, 3) for i in range(b)]
    SRblocks = torch.cat(SRblocks).permute(0, 2, 1)  # Mx4096*3
    HRblocks = torch.cat(HRblocks)  # Mx4096*3
    return SRlabels, SRblocks, HRblocks


def makeTestBlock(scene, target, crop_start):
    vrange = val_calcu(crop_start)
    b, _, _, _ = scene.shape
    no0index = (scene != -1.0).view(b, -1).float()  # BxN
    target = trans3D(scale(target), vrange)  # BxNx3
    scene = trans3D(scale(scene), vrange)  # BxNx3
    times = (torch.sum(no0index, 1) // 4096 + 1).int()  # B
    SRlabels = [torch.ones(times[i]) * i for i in range(b)]  # M
    SRlabels = torch.cat(SRlabels)
    sample_index = [
        torch.multinomial(no0index[i], times[i] * 4096, True) for i in range(b)
    ]  # Bx(times+1)
    SRblocks = [scene[i, sample_index[i], :].view(-1, 4096, 3) for i in range(b)]
    HRblocks = [target[i, sample_index[i], :].view(-1, 4096, 3) for i in range(b)]
    SRblocks = torch.cat(SRblocks).permute(0, 2, 1)  # Mx4096*3
    HRblocks = torch.cat(HRblocks)  # Mx4096*3
    return SRlabels, SRblocks, HRblocks


if __name__ == "__main__":
    a = torch.rand(25000, 3).uniform_(0, 120)
    imSR = torch.from_numpy(np.load("imSR.npy")).cuda()
    imHR = torch.from_numpy(np.load("imHR.npy")).cuda()

    HRscene = imHR[None, None, ...]
    SRscene = imSR[None, None, ...]
    vrange = val_calcu([384, 384])
    HRscene = trans3D(HRscene, vrange).squeeze(0)
    SRscene = trans3D(SRscene, vrange).squeeze(0)
    s2r = scene2blocks(4096, 15.0, 10.0, False, None, 1)
    HRblocks, SRblocks = s2r(HRscene, SRscene)
    print(HRblocks.shape, SRblocks.shape)


# if __name__ == "__main__":
#     vrange = val_calcu()
#     imSR = np.load("test/chamfer970/imSR.npy")
#     imSR = torch.FloatTensor(imSR)
#     XYZ = trans3D(imSR, vrange)
#     XYZ = np.asarray(XYZ)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(XYZ)
#     o3d.visualization.draw_geometries([pcd])

#     config = Dict(yaml.load(open(sys.argv[1]), Loader=yaml.SafeLoader))
#     device = get_device(config.cuda)

#     # Dataset
#     modal = config.modal
#     train_loader, val_loader = prepare_dataloader(config)

#     G = Generator(1, 1, 4)

#     # Tensorboard
#     exp_id = config.experiment_id
#     writer = SummaryWriter("runs/" + exp_id)
#     os.makedirs(osp.join("models", exp_id), exist_ok=True)
#     print("Experiment ID:", exp_id)

#     vrange = val_calcu()
#     colors_tensor = torch.randint(0, 255, [384, 384])
#     n_epoch = math.ceil(config.train.n_iter / len(train_loader))
#     for epoch in tqdm(range(1, n_epoch + 1), desc="Epoch"):
#         step = (epoch - 1) * len(train_loader)
#         # Validation
#         mse, ssim, psnr, summary, vertices_tensor = evaluator(
#             val_loader, G, device, config, step, vrange
#         )
#         writer.add_mesh(
#             "Point_clouds",
#             vertices=vertices_tensor,
#             colors=colors_tensor,
#             global_step=step,
#         )
#         writer.add_images("Results", summary, step)
#         writer.add_scalar("Score/MSE", mse, step)
#         writer.add_scalar("Score/SSIM", ssim, step)
#         writer.add_scalar("Score/PSNR", psnr, step)

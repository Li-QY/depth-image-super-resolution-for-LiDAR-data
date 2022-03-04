    def hook(name):
        def _hook(module, input, output):
            isnan = torch.any(torch.isnan(output[0]))
            print(name, isnan)

        return _hook

    for name, module in G.named_modules():
        module.register_forward_hook(hook(name))


    vrange = val_calcu(384)
    colors_tensor = torch.zeros([384, 384])
    colors_tensor = colors_tensor.view(-1)
    colors_tensor = colors_tensor[None, ...]
    colors_tensor = torch.cat((colors_tensor, colors_tensor, colors_tensor), dim=1)
    colors_tensor = colors_tensor[None, ...]

    
    
    self.model = net()
    checkpoint = torch.load('./model.pt')
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
    self.model.train()


    # loss_cham = criterion_cham(imgs_SR, imgs_HR_dep, sh, sw)
    # loss_cham = criterion_cham(imgs_SR, imgs_HR, sh, sw)
    # writer.add_mesh(
    #     "Point_clouds",
    #     vertices=vertices_tensor,
    #     colors=colors_tensor,
    #     global_step=step,
    # )

   # check
    imgs_LR_dep = F.interpolate(imgs_LR_dep, scale_factor=4)
    summary = [imgs_HR_ref, imgs_LR_dep, imgs_HR_dep, mask_HR]
    summary = torch.cat(summary, dim=2)[:8]
    summary = torch.clamp(scale(summary), 0.0, 1.0)
    torchvision.utils.save_image(summary, "check_aug.png")
    input()


    
    #calcu block number
        if not self.random_sample:
            num_block_x_po = (
                int(torch.ceil((limit_po[0] - self.block_size) / self.stride)) + 1
            )
            num_block_y_po = (
                int(torch.ceil((limit_po[1] - self.block_size) / self.stride)) + 1
            )
            num_block_x_ne = int(
                torch.ceil((limit_ne[0] - self.block_size) / self.stride) - 1
            )
            num_block_y_ne = int(
                torch.ceil((limit_ne[1] - self.block_size) / self.stride) - 1
            )
            xbeg_list, ybeg_list = torch.meshgrid(
                torch.arange(num_block_x_ne, num_block_x_po + 1),
                torch.arange(num_block_y_ne, num_block_y_po + 1),
            )
            xbeg_list = xbeg_list.reshape(-1) * self.stride
            ybeg_list = ybeg_list.reshape(-1) * self.stride
            return xbeg_list, ybeg_list, [limit_po, limit_ne]

        else:
            num_block_x = int(torch.ceil(limit_po[0] - limit_ne[0] / self.block_size))
            num_block_y = int(torch.ceil(limit_po[1] - limit_ne[1] / self.block_size))
            if self.sample_num is None:
                self.sample_num = num_block_x * num_block_y * self.sample_aug
            xbeg_list = torch.Tensor(self.sample_num).uniform_(
                limit_ne[0] - self.block_size, limit_po[0]
            )
            ybeg_list = torch.Tensor(self.sample_num).uniform_(
                limit_ne[1] - self.block_size, limit_po[1]
            )
            return xbeg_list, ybeg_list, [limit_po, limit_ne]
    
def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc

    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print(
                        "%s:%s%s %s"
                        % (
                            type(obj).__name__,
                            " GPU" if obj.is_cuda else "",
                            " pinned" if obj.is_pinned else "",
                            pretty_size(obj.size()),
                        )
                    )
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print(
                        "%s → %s:%s%s%s%s %s"
                        % (
                            type(obj).__name__,
                            type(obj.data).__name__,
                            " GPU" if obj.is_cuda else "",
                            " pinned" if obj.data.is_pinned else "",
                            " grad" if obj.requires_grad else "",
                            " volatile" if obj.volatile else "",
                            pretty_size(obj.data.size()),
                        )
                    )
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", total_size)


def adump_tensors(gpu_only=True):
    torch.cuda.empty_cache()
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    del obj
                    gc.collect()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    del obj
                    gc.collect()
        except Exception as e:
            pass
        
def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert isinstance(size, torch.Size)
    return " × ".join(map(str, size))



def hook(name):
    def _hook(module, input, output):
        isnan = torch.any(torch.isnan(output[0]))
        if isnan:
            print(name, isnan)
            breakpoint()

        return _hook


# for name, module in G.named_modules():
#     module.register_forward_hook(hook(name))


def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if N == num_sample:
        return data
    elif N > num_sample:
        sample = torch.multinomial(torch.arange(N).float(), num_sample).int()
        return data[sample.long(), :]
    else:
        sample = torch.multinomial(
            torch.arange(N).float(), num_sample - N, replacement=True
        )
        dup_data = data[sample.long(), :]
        return torch.cat([data, dup_data], 0)


def make_label(scene, target, crop_start):
    vrange = val_calcu(crop_start)
    target = trans3D(scale(target) * 120, vrange)  # BxNx3
    scene = trans3D(scale(scene) * 120, vrange)  # BxNx3
    b, _, _ = scene.shape
    s2b = scene2blocks(4096, 15.0, 10.0)
    blocks = []
    targets = []
    labels = []
    for i in range(b):
        HRscene = target[i][torch.sum(target[i].mul(target[i]), 1) != 0, :]
        SRscene = scene[i][torch.sum(scene[i].mul(scene[i]), 1) != 0, :]
        if (HRscene.shape[0] == 0) | (SRscene.shape[0] == 0):
            continue
        else:

            block_HR, block_SR = s2b(HRscene, SRscene)  # Nx3 -> mx4096x3
            label_img = torch.ones(block_SR.shape[0]) * i

            blocks.append(block_SR)
            targets.append(block_HR)
            labels.append(label_img)
    labels = torch.cat(labels)
    blocks = torch.cat(blocks).permute(0, 2, 1)  # Mx4096x3 -> Mx3x4096
    targets = torch.cat(targets)  # Mx4096x3
    return labels, blocks, targets


class scene2blocks:
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters
        stride: float, stride for block sweeping
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels
        
    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """

    def __init__(
        self,
        num_point,
        block_size=1.0,
        stride=1.0,
        random_sample=False,
        sample_num=None,
        sample_aug=1,
    ):
        self.num_point = num_point
        self.block_size = block_size
        self.stride = stride
        self.random_sample = random_sample
        self.sample_num = sample_num
        self.sample_aug = sample_aug

        assert self.stride <= self.block_size

    def HRcutlist(self, HRscene):
        limit_po = torch.max(HRscene, 0)[0]
        limit_ne = torch.min(HRscene, 0)[0]

        # Get the corner location for our sampling blocks
        if not self.random_sample:
            num_block_x_po = (
                int(torch.ceil((limit_po[0] - self.block_size) / self.stride)) + 1
            )
            num_block_y_po = (
                int(torch.ceil((limit_po[1] - self.block_size) / self.stride)) + 1
            )
            num_block_x_ne = int(
                torch.ceil((limit_ne[0] - self.block_size) / self.stride) - 1
            )
            num_block_y_ne = int(
                torch.ceil((limit_ne[1] - self.block_size) / self.stride) - 1
            )
            xbeg_list, ybeg_list = torch.meshgrid(
                torch.arange(num_block_x_ne, num_block_x_po + 1),
                torch.arange(num_block_y_ne, num_block_y_po + 1),
            )
            xbeg_list = xbeg_list.reshape(-1) * self.stride
            ybeg_list = ybeg_list.reshape(-1) * self.stride
            return xbeg_list, ybeg_list

        else:
            num_block_x = int(torch.ceil(limit_po[0] - limit_ne[0] / self.block_size))
            num_block_y = int(torch.ceil(limit_po[1] - limit_ne[1] / self.block_size))
            if self.sample_num is None:
                self.sample_num = num_block_x * num_block_y * self.sample_aug
            xbeg_list = torch.Tensor(self.sample_num).uniform_(
                limit_ne[0] - self.block_size, limit_po[0]
            )
            ybeg_list = torch.Tensor(self.sample_num).uniform_(
                limit_ne[1] - self.block_size, limit_po[1]
            )
            return xbeg_list, ybeg_list

    def collectblock(self, scene, idx, xbeg_list, ybeg_list, block_sizex, block_sizey):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        xcond = (scene[:, 0] <= xbeg + block_sizex) & (scene[:, 0] >= xbeg)
        ycond = (scene[:, 1] <= ybeg + block_sizey) & (scene[:, 1] >= ybeg)
        cond = xcond & ycond
        return cond

    def __call__(self, HRscene, SRscene):
        rate = (torch.max(SRscene, 0)[0] - torch.min(SRscene, 0)[0]) / (
            torch.max(HRscene, 0)[0] - torch.min(HRscene, 0)[0]
        )
        HRxlist, HRylist = self.HRcutlist(HRscene)
        SRxlist, SRylist = HRxlist * rate[0], HRylist * rate[1]
        SRblock_size = rate * self.block_size

        # Collect blocks
        HRblock_data_list = []
        SRblock_data_list = []
        idx = 0
        # i = 0
        for idx in range(len(HRxlist)):
            HRcond = self.collectblock(
                HRscene, idx, HRxlist, HRylist, self.block_size, self.block_size
            )
            SRcond = self.collectblock(
                SRscene, idx, SRxlist, SRylist, SRblock_size[0], SRblock_size[1]
            )
            if (torch.sum(HRcond) < 10) | (
                torch.sum(SRcond) < 10
            ):  # discard block if there are less than 100 pts.
                continue

            HRblock_data = HRscene[HRcond, :]
            SRblock_data = SRscene[SRcond, :]

            # randomly subsample data
            HRblock_data_sampled = sample_data(HRblock_data, self.num_point)
            SRblock_data_sampled = sample_data(SRblock_data, self.num_point)

            HRblock_data_list.append(HRblock_data_sampled[None, ...])  # list[1x4096x3]
            SRblock_data_list.append(SRblock_data_sampled[None, ...])  # list[1x4096x3]
            # i = i + 1
            # if i == 3:
            #     break

        HRblocks = torch.cat(HRblock_data_list)  # mx4096x3
        SRblocks = torch.cat(SRblock_data_list)  # mx4096x3
        return HRblocks, SRblocks

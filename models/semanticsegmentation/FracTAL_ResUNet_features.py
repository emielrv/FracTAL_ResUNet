import torch
import torch.nn.functional as F

from nn.layers.combine import combine_layers
from nn.layers.conv2Dnormed import Conv2DNormed
from nn.layers.scale import DownSample
from nn.pooling.psp_pooling import PSP_Pooling
from nn.units.fractal_resnet import FracTALResNet_unit


class FracTAL_ResUNet_features(torch.nn.Module):
    def __init__(self, nfilters_init, depth, widths=[1], psp_depth=4, verbose=True, norm_type='BatchNorm',
                 norm_groups=None, nheads_start=8, upFuse=False, ftdepth=5, **kwards):
        super().__init__(**kwards)

        if len(widths) == 1 and depth != 1:
            widths = widths * depth
        else:
            assert depth == len(widths), ValueError("depth and length of widths must match, aborting ...")

        self.conv_first = Conv2DNormed(nfilters_init, kernel_size=(1, 1), _norm_type=norm_type, norm_groups=norm_groups)

        self.convs_dn = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for idx in range(depth):
            nheads = nheads_start * 2 ** idx
            nfilters = nfilters_init * 2 ** idx
            if verbose:
                print(f"depth:= {idx}, nfilters: {nfilters}, nheads::{nheads}, widths::{widths[idx]}")
            tnet = torch.nn.Sequential()
            for _ in range(widths[idx]):
                tnet.add_module('FracTALResNet_unit',
                                FracTALResNet_unit(nfilters=nfilters, nheads=nheads, ngroups=nheads,
                                                   norm_type=norm_type, norm_groups=norm_groups, ftdepth=ftdepth))
            self.convs_dn.append(tnet)

            if idx < depth - 1:
                self.pools.append(DownSample(nfilters, _norm_type=norm_type, norm_groups=norm_groups))

        self.middle = PSP_Pooling(nfilters, depth=psp_depth, _norm_type=norm_type, norm_groups=norm_groups)

        self.convs_up = torch.nn.ModuleList()  # 1 argument
        self.UpCombs = torch.nn.ModuleList()  # 2 arguments
        for idx in range(depth - 1, 0, -1):
            nheads = nheads_start * 2 ** idx
            nfilters = nfilters_init * 2 ** (idx - 1)
            if verbose:
                print(f"depth:= {2 * depth - idx - 1}, nfilters: {nfilters}, nheads::{nheads}, widths::{widths[idx]}")

            tnet = torch.nn.Sequential()
            for _ in range(widths[idx]):
                tnet.add_module('FracTALResNet_unit',
                                FracTALResNet_unit(nfilters=nfilters, nheads=nheads, ngroups=nheads,
                                                   norm_type=norm_type, norm_groups=norm_groups, ftdepth=ftdepth))
            self.convs_up.append(tnet)

            ## No idea where this combine_layers_wthFusion comes from.. Surprised it worked in the original code, since it looks like the functino is missing
            if upFuse:
                self.UpCombs.append(combine_layers_wthFusion(nfilters=nfilters, nheads=nheads, _norm_type=norm_type,
                                                             norm_groups=norm_groups, ftdepth=ftdepth))
            else:
                self.UpCombs.append(combine_layers(nfilters, _norm_type=norm_type, norm_groups=norm_groups))

    def forward(self, input):
        conv1_first = self.conv_first(input)

        # ******** Going down ***************
        fusions = []
        pools = conv1_first

        for idx in range(self.depth):
            conv1 = self.convs_dn[idx](pools)
            if idx < self.depth - 1:
                # Evaluate fusions
                conv1 = conv1.clone()
                fusions = fusions + [conv1]
                # Evaluate pools
                pools = self.pools[idx](conv1)

        # Middle psppooling
        middle = self.middle(conv1)
        # Activation of middle layer
        middle = F.relu(middle)
        fusions = fusions + [middle]

        # ******* Coming up ****************
        convs_up = middle
        for idx in range(self.depth - 1):
            convs_up = self.UpCombs[idx](convs_up, fusions[-idx - 2])
            convs_up = self.convs_up[idx](convs_up)

        return convs_up, conv1_first

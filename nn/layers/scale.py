import torch
import torch.nn.functional as F

from nn.layers.conv2Dnormed import Conv2DNormed


class DownSample(torch.nn.Module):
    def __init__(self, nfilters, factor=2, _norm_type='BatchNorm', norm_groups=None, **kwargs):
        super().__init__()

        # Double the size of filters, since you downscale by 2. 
        self.factor = factor
        self.nfilters = nfilters * self.factor

        self.kernel_size = (3, 3)
        self.stride = (factor, factor)
        self.pad = (1, 1)

        with self.name_scope():
            self.convdn = Conv2DNormed(self.nfilters,
                                       kernel_size=self.kernel_size,
                                       stride=self.stride,
                                       padding=self.pad,
                                       _norm_type=_norm_type,
                                       norm_groups=norm_groups)

    def forward(self, x):
        x = self.convdn(x)
        return x


class UpSample(torch.nn.Module):
    def __init__(self, nfilters, factor=2, _norm_type='BatchNorm', norm_groups=None, **kwards):
        super().__init__()

        self.factor = factor
        self.nfilters = nfilters // self.factor

        with self.name_scope():
            self.convup_normed = Conv2DNormed(self.nfilters,
                                              kernel_size=(1, 1),
                                              _norm_type=_norm_type,
                                              norm_groups=norm_groups)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.factor, mode='nearest')
        x = self.convup_normed(x)
        return x
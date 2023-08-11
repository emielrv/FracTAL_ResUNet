import torch
import torch.nn.functional as F

from nn.layers.attention import FTAttention2D


class ResNet_v2_block(torch.nn.Module):
    def __init__(self, _nfilters, _kernel_size=(3, 3), _dilation_rate=(1, 1), _norm_type='BatchNorm', norm_groups=None,
                 ngroups=1, **kwards):
        super().__init__()

        self.nfilters = _nfilters
        self.kernel_size = _kernel_size
        self.dilation_rate = _dilation_rate

        with self.name_scope():
            # Ensures padding = 'SAME' for ODD kernel selection
            p0 = self.dilation_rate[0] * (self.kernel_size[0] - 1) // 2
            p1 = self.dilation_rate[1] * (self.kernel_size[1] - 1) // 2
            p = (p0, p1)

            self.BN1 = torch.nn.BatchNorm2d(self.nfilters,
                                            groups=ngroups) if _norm_type == 'BatchNorm' else torch.nn.GroupNorm(
                num_groups=norm_groups, num_channels=self.nfilters)
            self.conv1 = torch.nn.Conv2d(self.nfilters, self.nfilters, kernel_size=self.kernel_size, padding=p,
                                         dilation=self.dilation_rate, bias=False, groups=ngroups)
            self.BN2 = torch.nn.BatchNorm2d(self.nfilters,
                                            groups=ngroups) if _norm_type == 'BatchNorm' else torch.nn.GroupNorm(
                num_groups=norm_groups, num_channels=self.nfilters)
            self.conv2 = torch.nn.Conv2d(self.nfilters, self.nfilters, kernel_size=self.kernel_size, padding=p,
                                         dilation=self.dilation_rate, bias=True, groups=ngroups)

    def forward(self, _input_layer):
        x = self.BN1(_input_layer)
        x = F.relu(x)
        x = self.conv1(x)

        x = self.BN2(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x


class FracTALResNet_unit(torch.nn.Module):
    def __init__(self, nfilters, ngroups=1, nheads=1, kernel_size=(3, 3), dilation_rate=(1, 1), norm_type='BatchNorm',
                 norm_groups=None, ftdepth=5, **kwards):
        super().__init__()

        with self.name_scope():
            self.block1 = ResNet_v2_block(nfilters, kernel_size, dilation_rate, _norm_type=norm_type,
                                          norm_groups=norm_groups, ngroups=ngroups)
            self.attn = FTAttention2D(nkeys=nfilters, nheads=nheads, kernel_size=kernel_size, norm=norm_type,
                                      norm_groups=norm_groups, ftdepth=ftdepth)

            self.gamma = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, input):
        out1 = self.block1(input)

        att = self.attn(input)
        att = torch.mul(self.gamma, att)

        out = torch.mul(input + out1, torch.ones_like(out1) + att)
        return out

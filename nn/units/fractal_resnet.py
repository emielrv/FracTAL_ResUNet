import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.layers.attention import FTAttention2D

class ResNet_v2_block(nn.Module):
    def __init__(self, nfilters, kernel_size=(3, 3), dilation_rate=(1, 1), norm_type='BatchNorm', norm_groups=None,
                 ngroups=1, **kwargs):
        super().__init__()

        self.nfilters = nfilters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        # Ensures padding = 'SAME' for ODD kernel selection
        p0 = self.dilation_rate[0] * (self.kernel_size[0] - 1) // 2
        p1 = self.dilation_rate[1] * (self.kernel_size[1] - 1) // 2
        p = (p0, p1)

        if norm_type == 'BatchNorm':
            self.BN1 = nn.BatchNorm2d(self.nfilters, groups=ngroups)
        else:
            self.BN1 = nn.GroupNorm(num_groups=norm_groups, num_channels=self.nfilters)
        self.conv1 = nn.Conv2d(self.nfilters, self.nfilters, kernel_size=self.kernel_size, padding=p,
                               dilation=self.dilation_rate, bias=False, groups=ngroups)
        
        if norm_type == 'BatchNorm':
            self.BN2 = nn.BatchNorm2d(self.nfilters, groups=ngroups)
        else:
            self.BN2 = nn.GroupNorm(num_groups=norm_groups, num_channels=self.nfilters)
        self.conv2 = nn.Conv2d(self.nfilters, self.nfilters, kernel_size=self.kernel_size, padding=p,
                               dilation=self.dilation_rate, bias=True, groups=ngroups)

    def forward(self, input_layer):
        x = self.BN1(input_layer)
        x = F.relu(x)
        x = self.conv1(x)

        x = self.BN2(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x

class FracTALResNet_unit(nn.Module):
    def __init__(self, nfilters, ngroups=1, nheads=1, kernel_size=(3, 3), dilation_rate=(1, 1), norm_type='BatchNorm',
                 norm_groups=None, ftdepth=5, **kwargs):
        super().__init__()

        self.block1 = ResNet_v2_block(nfilters, kernel_size, dilation_rate, norm_type=norm_type,
                                      norm_groups=norm_groups, ngroups=ngroups)
        self.attn = FTAttention2D(nkeys=nfilters, nheads=nheads, kernel_size=kernel_size, norm=norm_type,
                                  norm_groups=norm_groups, ftdepth=ftdepth)

        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, input):
        out1 = self.block1(input)

        att = self.attn(input)
        att = torch.mul(self.gamma, att)

        out = torch.mul(input + out1, torch.ones_like(out1) + att)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.layers.conv2Dnormed import Conv2DNormed

class PSP_Pooling(nn.Module):
    def __init__(self, nfilters, depth=4, _norm_type='BatchNorm', norm_groups=None, mob=False):
        super(PSP_Pooling, self).__init__()

        self.nfilters = nfilters
        self.depth = depth

        self.convs = nn.ModuleList()
        for _ in range(depth):
            self.convs.append(Conv2DNormed(self.nfilters, kernel_size=(1, 1), padding=(0, 0), _norm_type=_norm_type,
                                           norm_groups=norm_groups))

        self.conv_norm_final = Conv2DNormed(channels=self.nfilters, kernel_size=(1, 1), padding=(0, 0),
                                            _norm_type=_norm_type, norm_groups=norm_groups)

    def HalfSplit(self, x):
        b = torch.split(x, x.size(2) // 2, dim=2)  # Split along the first dimension
        c1 = torch.split(b[0], x.size(3) // 2, dim=3)  # Split along the second dimension
        c2 = torch.split(b[1], x.size(3) // 2, dim=3)  # Split along the second dimension

        d11 = c1[0]
        d12 = c1[1]

        d21 = c2[0]
        d22 = c2[1]

        return [d11, d12, d21, d22]

    def QuarterStitch(self, Dss):
        temp1 = torch.cat((Dss[0], Dss[1]), dim=-1)
        temp2 = torch.cat((Dss[2], Dss[3]), dim=-1)
        result = torch.cat((temp1, temp2), dim=2)

        return result

    def HalfPooling(self, x):
        Ds = self.HalfSplit(x)

        Dss = []
        for x in Ds:
            Dss += [x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]]

        return self.QuarterStitch(Dss)

    def SplitPooling(self, x, depth):
        if depth == 1:
            return self.HalfPooling(x)
        else:
            D = self.HalfSplit(x)
            return self.QuarterStitch([self.SplitPooling(d, depth - 1) for d in D])

    def forward(self, input):
        p = [input]
        # 1st:: Global Max Pooling.
        p += [self.convs[0](torch.ones_like(input) * F.adaptive_max_pool2d(input, output_size=1))]
        p += [self.convs[d](self.SplitPooling(input, d)) for d in range(1, self.depth)]
        out = torch.cat(p, dim=1)
        out = self.conv_norm_final(out)

        return out

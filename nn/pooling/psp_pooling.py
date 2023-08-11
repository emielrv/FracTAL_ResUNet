import torch

from nn.layers.conv2Dnormed import Conv2DNormed


class PSP_Pooling(torch.nn.Module):
    def __init__(self, nfilters, depth=4, _norm_type='BatchNorm', norm_groups=None, mob=False):
        super(PSP_Pooling, self).__init__()

        self.nfilters = nfilters
        self.depth = depth

        self.convs = torch.nn.ModuleList()
        with self.name_scope():
            for _ in range(depth):
                self.convs.append(Conv2DNormed(self.nfilters, kernel_size=(1, 1), padding=(0, 0), _norm_type=_norm_type,
                                               norm_groups=norm_groups))

            self.conv_norm_final = Conv2DNormed(channels=self.nfilters, kernel_size=(1, 1), padding=(0, 0),
                                                _norm_type=_norm_type, norm_groups=norm_groups)

    def HalfSplit(self, F, _a):
        b = torch.split(_a, _a.size(2) // 2, dim=2)  # Split First dimension
        c1 = torch.split(b[0], _a.size(3) // 2, dim=3)  # Split 2nd dimension
        c2 = torch.split(b[1], _a.size(3) // 2, dim=3)  # Split 2nd dimension

        d11 = c1[0]
        d12 = c1[1]

        d21 = c2[0]
        d22 = c2[1]

        return [d11, d12, d21, d22]

    def QuarterStitch(self, F, _Dss):
        temp1 = torch.cat((_Dss[0], _Dss[1]), dim=-1)
        temp2 = torch.cat((_Dss[2], _Dss[3]), dim=-1)
        result = torch.cat((temp1, temp2), dim=2)

        return result

    def HalfPooling(self, F, _a):
        Ds = self.HalfSplit(F, _a)

        Dss = []
        for x in Ds:
            Dss += [F.broadcast_mul(torch.ones_like(x), F.adaptive_max_pool2d(x, output_size=1))]

        return self.QuarterStitch(F, Dss)

    def SplitPooling(self, F, _a, depth):
        if depth == 1:
            return self.HalfPooling(F, _a)
        else:
            D = self.HalfSplit(F, _a)
            return self.QuarterStitch(F, [self.SplitPooling(F, d, depth - 1) for d in D])

    def forward(self, _input):
        p = [_input]
        # 1st:: Global Max Pooling .
        p += [self.convs[0](torch.ones_like(_input) * F.adaptive_max_pool2d(_input, output_size=1))]
        p += [self.convs[d](self.SplitPooling(F, _input, d)) for d in range(1, self.depth)]
        out = torch.cat(p, dim=1)
        out = self.conv_norm_final(out)

        return out

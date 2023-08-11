import torch
import torch.nn.functional as F

from nn.layers.conv2Dnormed import Conv2DNormed
from nn.layers.scale import UpSample


class combine_layers(torch.nn.Module):
    def __init__(self, _nfilters, _norm_type='BatchNorm', norm_groups=None, **kwards):
        super().__init__()

        with self.name_scope():
            # This performs convolution, no BatchNormalization. No need for bias.
            self.up = UpSample(_nfilters, _norm_type=_norm_type, norm_groups=norm_groups)

            self.conv_normed = Conv2DNormed(channels=_nfilters, kernel_size=(1, 1), padding=(0, 0),
                                            _norm_type=_norm_type, norm_groups=norm_groups)

    def forward(self, _layer_lo, _layer_hi):
        up = self.up(_layer_lo)
        up = F.relu(up)
        x = torch.cat((up, _layer_hi), dim=1)
        x = self.conv_normed(x)
        return x

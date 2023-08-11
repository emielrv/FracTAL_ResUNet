import torch


class Conv2DNormed(torch.nn.Module):
    def __init__(self, channels, kernel_size, strides=(1, 1),
                 padding=(0, 0), dilation=(1, 1), activation=None,
                 weight_initializer=None, in_channels=0, _norm_type='BatchNorm', norm_groups=None, axis=1, groups=1,
                 **kwards):
        super().__init__()

        with self.name_scope():
            self.conv2d = torch.nn.Conv2d(in_channels, channels, kernel_size=kernel_size,
                                          stride=strides, padding=padding,
                                          dilation=dilation, bias=False,
                                          groups=groups)

            self.norm_layer = torch.nn.BatchNorm2d(channels) if _norm_type == 'BatchNorm' else torch.nn.GroupNorm(
                num_groups=norm_groups, num_channels=channels)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm_layer(x)
        return x

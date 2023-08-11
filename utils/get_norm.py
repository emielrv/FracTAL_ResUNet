import torch

def get_norm(name, axis=1, norm_groups=None):
    if name == 'BatchNorm':
        return torch.nn.BatchNorm2d(num_features=axis)
    elif name == 'InstanceNorm':
        return torch.nn.InstanceNorm2d(num_features=axis)
    elif name == 'LayerNorm':
        return torch.nn.LayerNorm(normalized_shape=axis)
    elif name == 'GroupNorm' and norm_groups is not None:
        return torch.nn.GroupNorm(num_groups=norm_groups, num_channels=axis)
    else:
        raise NotImplementedError
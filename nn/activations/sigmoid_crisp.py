import torch


class SigmoidCrisp(torch.nn.Module):
    def __init__(self, smooth=1.e-2, **kwards):
        super().__init__()

        self.smooth = smooth
        self.gamma = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, input):
        out = self.smooth + torch.sigmoid(self.gamma)
        out = 1.0 / out

        out = torch.mul(input, out)
        out = torch.sigmoid(out)
        return out

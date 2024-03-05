import torch
import torch.nn as nn
from torch.nn import Parameter


class RMSNorm(nn.Module):
    """
    https://arxiv.org/pdf/1910.07467.pdf
    """

    def __init__(self, d_model: int, p: float = 1.0):
        super(RMSNorm, self).__init__()

        self.weight = Parameter(torch.ones(1, 1, d_model), requires_grad=True)
        self.bias = Parameter(torch.zeros(1, 1, d_model), requires_grad=True)

    def forward(self, x):
        rms = self._rms(x)

        out = x / rms * self.weight + self.bias
        return out

    def _rms(self, x):
        return torch.mean(x ** 2, dim=-1, keepdim=True)

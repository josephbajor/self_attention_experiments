import torch
import torch.nn as nn
import torch.nn.functional as F

from hparams import Hparams


class FeedForward(nn.Module):
    """
    Standard feed forward network
    """

    def __init__(
        self, input_size: int, out_size: int, multiplier: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(input_size, input_size * multiplier),
            nn.GELU(),
            nn.Linear(input_size * multiplier, out_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

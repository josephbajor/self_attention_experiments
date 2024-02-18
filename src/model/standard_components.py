import torch
import torch.nn as nn
import torch.nn.functional as F

from hparams import Hparams


class FeedForward(nn.Module):
    """
    Standard feed forward network
    """

    def __init__(
        self, input_size: int, hidden_size: int, out_size: int, dropout: float
    ) -> None:
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GeLU(),
            nn.Linear(hidden_size, out_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)

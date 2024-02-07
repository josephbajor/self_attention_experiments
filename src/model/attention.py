import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import Hparams
from typing import Literal, Optional

from src.model.standard_components import FeedForward


class AttentionBlock(nn.Module):
    def __init__(
        self,
        attention_func: nn.Module,
        hparams: Hparams,
        emb_func: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.num_heads = hparams.num_heads
        self.block_size = hparams.att_block_size
        self.embed_size = hparams.embed_size
        self.ff_hidden_size = hparams.ff_hidden_size
        self.dropout = hparams.dropout

        self.attention_heads = nn.ModuleList(
            [attention_func(hparams, emb_func) for _ in range(self.num_heads)]
        )
        self.ff = FeedForward(
            input_size=self.num_heads * self.block_size,
            hidden_size=self.ff_hidden_size,
            out_size=self.embed_size,
            dropout=self.dropout,
        )

    def forward(self, x):
        x_out = torch.cat([layer(x) for layer in self.attention_heads], dim=-1)
        x_out = self.ff(x_out)

        return x_out


class FullAttention(nn.Module):
    def __init__(
        self,
        hparams: Hparams,
        emb_func: Optional[nn.Module] = None,
        masked: bool = False,
    ) -> None:
        super().__init__()

        self.max_seq_len = hparams.max_span
        self.block_size = hparams.att_block_size
        self.embed_size = hparams.embed_size

        self.q = nn.Linear(self.embed_size, self.block_size, bias=False)
        self.k = nn.Linear(self.embed_size, self.block_size, bias=False)
        self.v = nn.Linear(self.embed_size, self.block_size, bias=False)

        self.masked = masked
        if masked:
            self.register_buffer(
                "mask", torch.tril(torch.ones(self.max_seq_len, self.max_seq_len))
            )

        if emb_func is not None:
            self.emb = emb_func(hparams)
        else:
            self.emb = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        if self.emb is not None:
            q, k = self.emb(q, k)

        attention_values = q @ k.transpose(-2, -1)
        if self.masked:
            attention_values = attention_values.masked_fill(
                self.mask == 0, float("-inf")
            )

        attention_values = attention_values / np.sqrt(self.block_size)
        attention_values = F.softmax(
            attention_values / np.sqrt(self.block_size), dim=-1
        )

        z = attention_values @ v

        return z


class FnetAttention(nn.Module):
    def __init__(
        self,
        hparams: Hparams,
        emb_func: Optional[nn.Module] = None,
        masked: bool = False,
    ) -> None:
        super().__init__()

        self.max_seq_len = hparams.max_span
        self.embed_size = hparams.embed_size
        self.block_size = hparams.att_block_size

        self.q = nn.Linear(self.embed_size, self.block_size, bias=False)
        self.k = nn.Linear(self.embed_size, self.block_size, bias=False)

        if masked:
            self.register_buffer(
                "mask", torch.tril(torch.ones(self.max_seq_len, self.max_seq_len))
            )
        else:
            self.mask = None

        if emb_func is not None:
            self.emb = emb_func(hparams)
        else:
            self.emb = None

    @torch.cuda.amp.autocast(enabled=False)
    def fft_fwd(self, hidden):
        return torch.fft.fftn(hidden).real

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)

        if self.emb is not None:
            q, k = self.emb(q, k)

        attention_values_q = self.fft_fwd(q)
        attention_values_k = self.fft_fwd(k)

        return attention_values_q + attention_values_k


ATT_FUNC_MAP = {
    "full": FullAttention,
    "fnet": FnetAttention,
}

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
        self.embed_size = hparams.embed_size
        self.dropout = hparams.dropout
        self.ff_mult = hparams.ff_internal_mult

        self.ln_1 = nn.LayerNorm(self.embed_size)
        self.attention = attention_func(hparams, emb_func)
        self.ln_2 = nn.LayerNorm(self.embed_size)

        self.ff = FeedForward(
            input_size=self.embed_size,
            dropout=self.dropout,
            multiplier=self.ff_mult,
        )

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))

        return x


class FullAttention(nn.Module):
    def __init__(
        self,
        hparams: Hparams,
        emb_func: Optional[nn.Module] = None,
        masked: bool = True,
    ) -> None:
        super().__init__()
        if hparams.embed_size % hparams.num_heads != 0:
            raise ValueError(
                f"Embed size {hparams.embed_size} must be divisible by num_heads {hparams.num_heads}"
            )

        self.max_seq_len = hparams.max_span
        self.embed_size = hparams.embed_size
        self.use_flash = hparams.use_flash
        self.n_heads = hparams.num_heads
        self.dropout = hparams.dropout

        self.attention_components = nn.Linear(self.embed_size, 3 * self.embed_size)

        self.output_projection = nn.Linear(self.embed_size, self.embed_size)
        self.output_dropout = nn.Dropout(self.dropout)

        self.masked = masked
        if masked:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(self.max_seq_len, self.max_seq_len)).view(
                    1, 1, self.max_seq_len, self.max_seq_len
                ),
            )
        else:
            self.mask = None

        if emb_func is not None:
            self.emb = emb_func(hparams)
        else:
            self.emb = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = (
            x.shape
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        q, k, v = self.attention_components(x).split(self.embed_size, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if self.emb is not None:
            q, k = self.emb(q, k)

        if self.use_flash:
            z = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=self.masked,
                dropout_p=self.dropout if self.training else 0,
            )

        else:
            attention_values = q @ k.transpose(-2, -1)
            if self.masked:
                attention_values = attention_values.masked_fill(
                    self.mask[:, :, :T, :T] == 0, float("-inf")
                )

            attention_values = F.softmax(
                attention_values / np.sqrt(q.shape[-1]), dim=-1
            )

            z = attention_values @ v

        z = z.transpose(1, 2).contiguous().view(B, T, C)
        z = self.output_dropout(self.output_projection(z))

        return z


class FnetAttention(nn.Module):
    def __init__(
        self,
        hparams: Hparams,
        emb_func: Optional[nn.Module] = None,
        masked: bool = True,
    ) -> None:
        super().__init__()

        self.max_seq_len = hparams.max_span
        self.embed_size = hparams.embed_size

        self.q = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.k = nn.Linear(self.embed_size, self.embed_size, bias=False)

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

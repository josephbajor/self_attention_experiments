import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import Hparams

from src.utils import generate_rand_emb


class AttentionFunc:
    def __init__(self) -> None:
        pass


class SimpleBigramModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()

        self.embed = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.embed(x)

    def generate(self, x, seq_len, deterministic=False):
        for seq in range(seq_len):
            # logits = self(x)[:, -1, :]
            logits = self(x)[-1, :]  # non-batched
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                next_idx = torch.argmax(dim=1)
            else:
                next_idx = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, next_idx), dim=-1)

        return x


class FullAttention(nn.Module):
    def __init__(
        self, max_seq_len, embed_size, block_size, masked=False, *args, **kwargs
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.block_size = block_size

        self.q = nn.Linear(embed_size, block_size, bias=False)
        self.k = nn.Linear(embed_size, block_size, bias=False)
        self.v = nn.Linear(embed_size, block_size, bias=False)

        self.masked = masked
        if masked:
            self.register_buffer(
                "mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
            )

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

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


class FnetAttentionWithBinaryPosEmb(nn.Module):
    def __init__(
        self,
        max_seq_len,
        embed_size,
        block_size,
        device,
        universal=False,
        masked=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.q = nn.Linear(embed_size, block_size, bias=False)
        self.k = nn.Linear(embed_size, block_size, bias=False)

        if masked:
            self.register_buffer(
                "mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
            )

        # self.binary_emb = BinaryPosEncoding(
        #     block_size=block_size, device=device, universal=universal
        # )

        self.binary_emb = StaticBinaryPosEncoding(
            block_size=block_size, max_span=max_seq_len, device=device
        )

    @torch.cuda.amp.autocast(enabled=False)
    def fft_fwd(self, hidden):
        return torch.fft.fftn(hidden).real

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)

        q_emb, k_emb = self.binary_emb(q, k)

        attention_values_q = self.fft_fwd(q_emb)
        attention_values_k = self.fft_fwd(k_emb)

        return attention_values_q + attention_values_k


class AttentionBlock(nn.Module):
    def __init__(
        self,
        attention_func: tuple[nn.Module, dict],
        max_seq_len: int,
        embed_size: int,
        block_size: int,
        num_heads: int,
        out_size: int,
        device: str,
        masked: bool = False,
    ):
        super().__init__()

        self.attention_heads = nn.ModuleList(
            [attention_func[0](**attention_func[1]) for _ in range(num_heads)]
        )
        self.lin = nn.Linear(block_size * num_heads, out_size)

    def forward(self, x):
        x_out = torch.cat([layer(x) for layer in self.attention_heads], dim=-1)
        x_out = self.lin(x_out)
        x_out = F.relu(x_out)

        return x_out


class StaticBinaryPosEncoding(nn.Module):
    def __init__(self, block_size, max_span, device: str) -> None:
        super().__init__()
        self.device = device

        self.emb = torch.tensor(generate_rand_emb(block_size, max_span)).to(device)

    def forward(self, q, k):
        if len(q.shape) > 2:
            emb = self.emb.repeat(q.shape[0], 1, 1)
        else:
            emb = self.emb
        q_out = q + emb
        k_out = k + emb

        return q_out, k_out


class BinaryPosEncoding(nn.Module):
    def __init__(self, block_size, device: str, universal: bool = False) -> None:
        super().__init__()
        self.device = device
        self.universal = universal

        self.E1 = nn.Linear(block_size, block_size, bias=False)
        if not self.universal:
            self.E2 = nn.Linear(block_size, block_size, bias=False)

    def clip_grad(self):
        # Might be possible to detatch weights from the compute graph and
        # manually update them by rounding
        # Might not work with autograd
        pass

    def forward(self, q, k):
        self.index = torch.arange(
            q.shape[-1], device=self.device, dtype=torch.float32
        ).repeat(q.shape[-2], 1)

        e1_embedding = torch.round(F.sigmoid(self.E1(self.index)))

        q_pos = q + e1_embedding
        if not self.universal:
            e2_embedding = (F.sigmoid(self.E2(self.index)) > 0.5).float()
            k_pos = k + e2_embedding
        else:
            k_pos = k + e1_embedding

        return q_pos, k_pos


class AttentionLM(nn.Module):
    def __init__(
        self,
        hparams: Hparams,
        vocab_size,
        device,
        attention_pos_emb_func: bool = False,
    ) -> None:
        super().__init__()

        # self.vocab_size = vocab_size
        # self.embed_size = hparams.embed_size
        self.use_positional_embedding = hparams.use_positional_embedding

        # Network Components
        self.device = device
        self.embed = nn.Embedding(vocab_size, hparams.embed_size)
        if self.use_positional_embedding:
            self.embed_pos = nn.Embedding(hparams.max_span, hparams.embed_size)
        self.max_span = hparams.max_span

        self.attention = AttentionBlock(
            attention_func=(
                FnetAttentionWithBinaryPosEmb,
                {
                    "max_seq_len": hparams.max_span,
                    "embed_size": hparams.embed_size,
                    "block_size": hparams.att_block_size,
                    "device": self.device,
                    "universal": False,
                    "masked": False,
                },
            ),
            max_seq_len=hparams.max_span,
            embed_size=hparams.embed_size,
            block_size=hparams.att_block_size,
            num_heads=hparams.num_heads,
            out_size=vocab_size,
            device=self.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)

        if self.use_positional_embedding:
            pos_emb = self.embed_pos(torch.arange(x.shape[-1], device=self.device))
            x = emb + pos_emb

        x = F.gelu(x)

        logits = self.attention(x)

        return logits

    def generate_batch(self, x, seq_len, deterministic=False):
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)

        for seq in range(seq_len):
            input = x[:, -self.max_span :]
            logits = self(input)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                next_idx = torch.argmax(input=probs, dim=1)
            else:
                next_idx = torch.multinomial(input=probs, num_samples=1)

            next_idx = next_idx.unsqueeze(dim=0)
            x = torch.cat((x, next_idx), dim=-1)

        return x

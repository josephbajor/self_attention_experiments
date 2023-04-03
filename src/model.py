import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import Hparams


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
    def __init__(self, max_seq_len, embed_size, block_size, autoregressive=False):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.q = nn.Linear(embed_size, block_size, bias=False)
        self.k = nn.Linear(embed_size, block_size, bias=False)
        self.v = nn.Linear(embed_size, block_size, bias=False)

        self.autoregressive = autoregressive
        if autoregressive:
            self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len))

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attention_values = q @ k.transpose(-2, -1)
        if self.autoregressive:
            attention_values = attention_values.masked_fill(
                self.mask == 0, float("-inf")
            )

        attention_values = attention_values / torch.sqrt(self.block_size)
        attention_values = F.softmax(
            attention_values / torch.sqrt(self.block_size), dim=-1
        )

        z = attention_values @ v

        return z


class AttentionLM(nn.Module):
    def __init__(self, hparams: Hparams, vocab_size) -> None:
        super().__init__()

        # self.vocab_size = vocab_size
        # self.embed_size = hparams.embed_size

        # Network Components
        self.embed = nn.Embedding(vocab_size, hparams.embed_size)
        self.embed_pos = nn.Embedding(hparams.max_span, hparams.embed_size)
        self.l1 = nn.Linear(hparams.embed_size, vocab_size)

    def forward(self, x):
        emb = self.embed(x)
        pos_emb = self.embed_pos(torch.arange(x.shape[-1], device="cuda"))

        x = emb + pos_emb
        x = F.gelu(x)

        logits = self.l1(x)

        return logits

    def generate_batch(self, x, seq_len, deterministic=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=0)

        for seq in range(seq_len):
            logits = self(x)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                next_idx = torch.argmax(dim=1)
            else:
                next_idx = torch.multinomial(input=probs, dim=1, num_samples=1)

            x = torch.cat((x, next_idx), dim=1)

        return x

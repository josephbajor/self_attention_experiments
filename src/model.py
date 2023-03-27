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


class AttentionLM(nn.Module):
    def __init__(self, hparams: Hparams, vocab_size) -> None:
        super().__init__()

        # self.vocab_size = vocab_size
        # self.embed_size = hparams.embed_size

        # Network Components
        self.embed = nn.Embedding(vocab_size, hparams.embed_size)
        self.embed_pos = nn.Embedding(hparams.block_size, hparams.embed_size)
        self.l1 = nn.Linear(hparams.embed_size, vocab_size)

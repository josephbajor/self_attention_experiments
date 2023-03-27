import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBigramModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()

        self.embed = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.embed(x)

    def generate(self, x, seq_len, deterministic=False):
        for seq in range(seq_len):
            logits = self(x)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                next_idx = torch.argmax(dim=1)
            else:
                next_idx = torch.multinomial(dim=1)

            x = torch.cat((x, next_idx), dim=1)

        return x

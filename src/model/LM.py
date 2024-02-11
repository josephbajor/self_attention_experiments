import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import Hparams
from typing import Literal, Optional

from src.model.attention import ATT_FUNC_MAP, AttentionBlock
from src.model.encoding import ENC_FUNC_MAP


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
    def __init__(
        self,
        hparams: Hparams,
        vocab_size: int,
    ) -> None:
        super().__init__()

        self.use_positional_embedding = hparams.use_positional_embedding

        # Network Components
        self.embed = nn.Embedding(vocab_size, hparams.embed_size)
        if self.use_positional_embedding:
            self.embed_pos = nn.Embedding(hparams.max_span, hparams.embed_size)
        self.max_span = hparams.max_span

        att_func_type = hparams.att_func_type
        emb_func = hparams.emb_func

        self.attention = nn.ModuleList(
            [
                AttentionBlock(
                    attention_func=ATT_FUNC_MAP[att_func_type],
                    hparams=hparams,
                    emb_func=ENC_FUNC_MAP[emb_func] if emb_func is not None else None,
                )
                for _ in range(hparams.att_layers)
            ]
        )

        self.linear = nn.Linear(hparams.embed_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)

        if self.use_positional_embedding:
            pos_emb = self.embed_pos(torch.arange(x.shape[-1], device=x.device))
            x = emb + pos_emb

        x = F.gelu(x)

        for layer in self.attention:
            x = layer(x)

        x = self.linear(x)

        return x

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

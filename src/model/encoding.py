import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import Hparams
from src.utils import generate_rand_emb


class StaticBinaryPosEncoding(nn.Module):
    def __init__(self, hparams: Hparams) -> None:
        super().__init__()

        self.head_size = hparams.embed_size // hparams.num_heads

        self.emb = torch.tensor(generate_rand_emb(self.head_size, hparams.max_span))

    def forward(self, q, k):
        # move emb to current device if necessary
        if self.emb.device != q.device:
            self.emb = self.emb.to(q.device)

        if len(q.shape) > 2:
            emb = self.emb.repeat(q.shape[0], 1, 1)
        else:
            emb = self.emb

        if self.emb.device != q.device:
            self.emb = emb.to(q.device)

        q_out = q + emb
        k_out = k + emb

        return q_out, k_out


class BinaryPosEncoding(nn.Module):
    def __init__(self, hparams: Hparams) -> None:
        super().__init__()
        self.universal = hparams.universal_pos_enc

        self.head_size = hparams.embed_size // hparams.num_heads

        self.E1 = nn.Linear(self.head_size, self.head_size, bias=False)
        if not self.universal:
            self.E2 = nn.Linear(self.head_size, self.head_size, bias=False)

    def forward(self, q, k):
        self.index = torch.arange(
            q.shape[-1], dtype=torch.float32, device=q.device
        ).repeat(q.shape[-2], 1)

        e1_embedding = torch.round(F.sigmoid(self.E1(self.index)))

        q_pos = q + e1_embedding
        if not self.universal:
            e2_embedding = torch.round(F.sigmoid(self.E2(self.index)))
            k_pos = k + e2_embedding
        else:
            k_pos = k + e1_embedding

        return q_pos, k_pos


ENC_FUNC_MAP = {
    "binary_static": StaticBinaryPosEncoding,
    "binary_learned": BinaryPosEncoding,
}

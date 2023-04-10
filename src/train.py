import numpy as np
import pickle
import torch
import torch.nn.functional as F

import os
import glob

from tqdm import tqdm

from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from src.model import SimpleBigramModel, AttentionLM
from src.dataloaders import build_loaders
from hparams import Hparams


def train():
    hparams = Hparams()

    tokenizer, train_loader, _, _ = build_loaders(hparams)

    model = AttentionLM(hparams, vocab_size=tokenizer.get_vocab_size())
    model = model.to("cuda")

    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters())

    model.train()
    for epoch in range(hparams.epochs):
        for x, y in tqdm(train_loader):
            optim.zero_grad()

            x = x.to("cuda")
            y = y.to("cuda")

            logits = model(x)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            y = y.view(B * T)

            loss = loss_fn(logits, y)

            loss.backward()
            optim.step()

        print(loss)

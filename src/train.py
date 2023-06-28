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
from src.inference import inference
from hparams import Hparams



def train(device = 'cuda'):
    hparams = Hparams()

    tokenizer, train_loader, _, _ = build_loaders(hparams)

    model = AttentionLM(hparams, vocab_size=tokenizer.get_vocab_size(), device = device)
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters())

    model.train()
    for epoch in range(hparams.epochs):
        for step, (x, y) in tqdm(enumerate(train_loader)):
            optim.zero_grad()

            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            y = y.view(B * T)

            loss = loss_fn(logits, y)

            loss.backward()
            optim.step()

            if step%3000 == 0:
                tests = [
                    'Hello ',
                    'The',
                    'W',
                ]
                
                print(f'generation test | step {step}:')

                for test in tests:
                    test_gen = inference(test, model, tokenizer, out_len=60, determenistic=True, device=device)
                    print(test_gen)

                
            if step%10000 == 0:
                torch.save(model.state_dict(), hparams.tokenized_dir + '/model.pth')
                

        print(loss)

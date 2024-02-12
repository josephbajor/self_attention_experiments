import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torchinfo import summary

import os
import glob
import uuid
from pathlib import Path
from src import logger

from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from src.model.LM import SimpleBigramModel, AttentionLM
from src.dataloaders import build_loaders
from src.inference import inference
from hparams import Hparams


def train(device="cuda"):
    hparams = Hparams()

    tokenizer, train_loader, val_loader, _ = build_loaders(hparams)
    logger.info(f"Vocab Size: {tokenizer.get_vocab_size()}")
    logger.info(f"Train Size: {len(train_loader.dataset)}")
    logger.info(f"Val Size: {len(val_loader.dataset)}")
    if hparams.eval_steps is not None:
        logger.info(f"clipping val loader to {hparams.eval_steps} samples")
        val_loader = Subset(
            val_loader, torch.randperm(len(val_loader))[: hparams.eval_steps]
        ).dataset

    model = AttentionLM(hparams, vocab_size=tokenizer.get_vocab_size())
    logger.info(f"Loading model to device: {device}")
    model = model.to(device)
    if hparams.compile_model:
        logger.info("Compiling model...")
        model.compile()
        logger.info("Model compiled")

    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters())

    x_samp, _ = train_loader.dataset[0]
    summary(model, input_data=x_samp.to(device))

    loss_buffer = hparams.windowed_loss_buffer_size
    save_pth = f"checkpoints/{str(uuid.uuid4())[:8]}"
    Path(f"{save_pth}").mkdir(exist_ok=True, parents=True)
    hparams.save_to_file(f"{save_pth}/hparams.json")

    logger.info(f"model params: {model.get_param_count()}")

    model.train()
    for epoch in range(hparams.epochs):
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            TextColumn("[bold blue]{task.fields[loss]}"),
            transient=False,
        ) as progress:
            windowed_loss = np.zeros(loss_buffer, dtype=np.float32)
            train_task = progress.add_task(
                "[green]Training...", total=len(train_loader), loss="Initializing..."
            )

            for step, (x, y) in enumerate(train_loader):

                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                y = y.view(B * T)

                loss = loss_fn(logits, y)

                loss.backward()
                optim.step()

                windowed_loss[step % loss_buffer] = loss

                progress.update(
                    train_task,
                    advance=1,
                    loss=f"Train Loss: {windowed_loss.mean():.5f}",
                )
                optim.zero_grad(set_to_none=True)

                if step % 1000 == 0:
                    # evaluate model
                    model.eval()
                    val_loss = 0
                    val_task = progress.add_task(
                        "[cyan]Validating...",
                        total=len(val_loader),
                        loss="Initializing...",
                    )
                    for val_step, (x, y) in enumerate(val_loader):
                        x = x.to(device)
                        y = y.to(device)

                        logits = model(x)
                        B, T, C = logits.shape
                        logits = logits.view(B * T, C)
                        y = y.view(B * T)

                        loss = loss_fn(logits, y)
                        val_loss += loss.item()
                        progress.update(
                            val_task, advance=1, loss=f"Val Loss: {loss/val_step:.5f}"
                        )
                    progress.remove_task(val_task)
                    logger.info(f"Validation Loss: {val_loss / hparams.eval_steps:.5f}")

                    # generation test
                    tests = [
                        "Hello ",
                        "The",
                        "W",
                    ]

                    logger.info(f"generation test | step {step}:")

                    for test in tests:
                        test_gen = inference(
                            test,
                            model,
                            tokenizer,
                            out_len=20,
                            determenistic=False,
                            device=device,
                        )
                        logger.info(f"{test} -> {test_gen}")

                    # put model back into training mode
                    model.train()

                if step % 10000 == 0:
                    # save training checkpoint
                    logger.info(f"Saving checkpoint at step {step} to {save_pth}")
                    torch.save(model.state_dict(), f"{save_pth}/model.pth")

            logger.info(loss)

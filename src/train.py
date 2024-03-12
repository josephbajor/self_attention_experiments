import numpy as np
import torch
import torch.nn.functional as F
from torchinfo import summary
import wandb

import uuid
from pathlib import Path
from src import logger, console

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    MofNCompleteColumn,
)

from src.model.LM import AttentionLM
from src.dataloaders import build_loaders, build_loaders_shakespeare
from src.inference import inference
from src.utils import RateColumn, initiate_run
from hparams import Hparams


def train(device="cuda"):
    hparams = Hparams()

    logger.info(f"Training on device: {device}")

    if hparams.dataset == "wikitext":
        tokenizer, train_loader, val_loader, _ = build_loaders(hparams)
    elif hparams.dataset == "shakespeare":
        tokenizer, train_loader, val_loader = build_loaders_shakespeare(hparams)

    logger.info(f"Vocab Size: {tokenizer.get_vocab_size()}")
    logger.info(f"Train Size: {len(train_loader)}")
    logger.info(f"Val Size: {len(val_loader)}")

    if hparams.type == "gpt":
        model = AttentionLM(hparams, vocab_size=tokenizer.get_vocab_size())
    if hparams.type == "nanogpt":
        from model import GPTConfig, GPT

        gpt_args = dict(
            n_layer=hparams.att_layers,
            n_head=hparams.num_heads,
            n_embd=hparams.embed_size,
            block_size=hparams.max_span,
            vocab_size=tokenizer.get_vocab_size(),
        )
        model = GPT(GPTConfig(**gpt_args))

    logger.info(f"Loading model to device: {device}")
    model = model.to(device)

    # print model summary
    # we do this before compilation to avoid the prehooks screwing up torchdynamo
    summary(
        model,
        input_size=(hparams.batch_size, hparams.max_span),
        device=device,
        dtypes=[torch.long],
    )
    print(model)

    if hparams.compile_model:
        logger.info("Compiling model...")
        model.compile(dynamic=True)
        logger.info("Model compiled")

    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), fused=True, lr=hparams.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=hparams.half_precision)

    loss_buffer = hparams.windowed_loss_buffer_size
    save_pth = f"checkpoints/{str(uuid.uuid4())[:8]}"
    Path(f"{save_pth}").mkdir(exist_ok=True, parents=True)
    hparams.save_to_file(f"{save_pth}/hparams.json")

    if hparams.type == "gpt":
        logger.info(f"model params: {model.get_param_count()}")

    run = initiate_run(hparams, model)

    model.train()
    for epoch in range(hparams.epochs):
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            RateColumn(),
            MofNCompleteColumn(),
            TextColumn("[bold blue]{task.fields[loss]}"),
            transient=True,
            console=console,
        ) as progress:
            windowed_loss = np.zeros(loss_buffer, dtype=np.float32)
            train_task = progress.add_task(
                "[green]Training...", total=len(train_loader), loss="Initializing..."
            )

            for step, (x, y) in enumerate(train_loader):

                x = x.to(device)
                y = y.to(device)

                with torch.cuda.amp.autocast(enabled=hparams.half_precision):
                    if hparams.type == "nanogpt":
                        logits, loss = model(x, y)

                    elif hparams.type == "gpt":
                        logits = model(x)
                        loss = loss_fn(logits.transpose(-1, -2), y)

                scaler.scale(loss).backward()
                # clip gradients
                if hparams.gradient_clip_val != 0.0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), hparams.gradient_clip_val
                    )
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

                windowed_loss[step % loss_buffer] = loss

                progress.update(
                    train_task,
                    advance=1,
                    loss=f"Train Loss: {windowed_loss[:step+1].mean():.5f}",
                )

                if step % hparams.eval_every_n_steps == 0 and step > 0:
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

                        if hparams.type == "nanogpt":
                            logits, loss = model(x, y)
                        elif hparams.type == "gpt":
                            logits = model(x)
                            B, T, C = logits.shape
                            logits = logits.view(B * T, C)
                            y = y.view(B * T)

                            loss = loss_fn(logits, y)

                        val_loss += loss.item()
                        progress.update(
                            val_task,
                            advance=1,
                            loss=f"Val Loss: {val_loss / (val_step + 1):.5f}",
                        )
                    progress.remove_task(val_task)
                    logger.info(f"Validation Loss: {val_loss / (val_step + 1):.5f}")

                    wandb.log(
                        {
                            "train_loss": windowed_loss[: step + 1].mean(),
                            "val_loss": val_loss / (val_step + 1),
                            "learning_Rate": optim.param_groups[0]["lr"],
                        }
                    )

                    # generation test
                    tests = ["Hello ", "The", "W", "This is ", "Wher"]

                    logger.info(f"generation test | step {step}:")

                    for test in tests:
                        test_gen = inference(
                            test,
                            model,
                            tokenizer,
                            out_len=80,
                            determenistic=False,
                            device=device,
                            mode=hparams.dataset,
                        )
                        logger.info(f"{test} -> {test_gen}")

                    # put model back into training mode
                    model.train()

                if step % 3000 == 0 and step > 0:
                    # save training checkpoint
                    logger.info(f"Saving checkpoint at step {step} to {save_pth}")
                    torch.save(model.state_dict(), f"{save_pth}/model.pth")
                    hparams.save_to_file(f"{save_pth}/hparams.json")

            logger.info(loss)

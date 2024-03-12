import numpy as np
from rich.progress import ProgressColumn, Task, filesize
from rich.text import Text
from hparams import Hparams
import wandb
import torch
import time
import os


def generate_rand_emb(x, y):
    rng = np.random.default_rng(seed=42)

    # generate initial set
    e = {tuple(rng.binomial(1, 0.5, x)) for i in range(y)}

    while len(e) < y:
        e.add(rng.binomial(1, 0.5, x))

    return np.array(tuple(e))


class RateColumn(ProgressColumn):
    """Renders human readable processing rate."""

    def render(self, task: "Task") -> Text:
        """Render the speed in iterations per second."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("", style="progress.percentage")
        unit, suffix = filesize.pick_unit_and_suffix(
            int(speed),
            ["", "×10³", "×10⁶", "×10⁹", "×10¹²"],
            1000,
        )
        data_speed = speed / unit
        return Text(f"{data_speed:.1f}{suffix} it/s", style="progress.percentage")


def initiate_run(hparams: Hparams, model: torch.nn.Module):
    """
    Initialize connection to wandb and begin the run using provided hparams
    """
    # load wandb key from local environment
    # api_key = os.getenv("WANDB_API_KEY")
    api_key = "c319fb8dfa7ce22e07aa0cefe0823a9752d50720"

    wandb.login(key=api_key)

    if hparams.use_wandb:
        mode = "online"
    else:
        mode = "disabled"

    run = wandb.init(
        name=f"{hparams.architecture}_{int(time.time())}",
        project=hparams.project,
        config=hparams.to_dict(),
        mode=mode,
    )

    wandb.watch(model, log="gradients", log_freq=500)

    wandb.config.update(  # Add model parameter count
        {"parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    )

    return run

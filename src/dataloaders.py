import torch
import numpy as np
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE

from src.preprocessing import preprocess_wikitext
from hparams import Hparams


class Wiki103Dataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size) -> None:
        super().__init__()

        self.data = data
        self.block_size = block_size

        # set dataset length
        # reduce by block_size + 1 to account for loader lookahead and y shift
        self.len = len(data) - (block_size + 1)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.len


def build_loaders(hparams: Hparams):
    # Load data from disk
    data_path = f"{hparams.tokenized_dir}/wikitext_tokenized.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "data not found in {datapath}!\nRun preprocessing.py before training"
        )
    data = np.load(data_path)

    tokenizer_path = f"{hparams.tokenized_dir}/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            "tokenizer not found in {datapath}!\nRun preprocessing.py before training"
        )
    tokenizer = Tokenizer.from_file(tokenizer_path)

    train_set = Wiki103Dataset(data["train"], block_size=hparams.block_size)
    val_set = Wiki103Dataset(data["val"], block_size=hparams.block_size)
    test_set = Wiki103Dataset(data["test"], block_size=hparams.block_size)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=False,
    )
    return tokenizer, train_loader, val_loader, test_loader

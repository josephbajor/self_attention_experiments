import torch
import numpy as np
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE

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


class ShakespeareDataset(torch.utils.data.Dataset):
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


class TokenizerDummyWrapper:
    """
    Wrapper that takes the idx2char and char2idx from the Tokenizer and implements
    the encode and decode methods to be compatible with the inference function.
    """

    def __init__(self, idx2char, char2idx):
        self.idx2char = idx2char
        self.char2idx = char2idx

    def encode(self, text):
        return torch.tensor([self.char2idx[c] for c in text], dtype=torch.long)

    def decode(self, ids):
        return "".join([self.idx2char[i] for i in ids])

    def get_vocab_size(self):
        return len(self.idx2char)


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

    train_set = Wiki103Dataset(data["train"], block_size=hparams.max_span)
    val_set = Wiki103Dataset(data["val"], block_size=hparams.max_span)
    test_set = Wiki103Dataset(data["test"], block_size=hparams.max_span)

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


def build_loaders_shakespeare(hparams: Hparams):
    # Load data from disk
    data_path = f"{hparams.tokenized_dir}/shakespeare.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "data not found in {datapath}!\nRun preprocessing.py before training"
        )
    data = np.load(data_path, allow_pickle=True)

    train_set = ShakespeareDataset(data["train"], block_size=hparams.max_span)
    val_set = ShakespeareDataset(data["val"], block_size=hparams.max_span)

    if hparams.eval_steps is not None:
        val_set = torch.utils.data.Subset(
            val_set,
            torch.randperm(len(val_set))[: hparams.eval_steps * hparams.batch_size],
        )

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

    # build dummy tokenizer
    tokenizer = TokenizerDummyWrapper(data["idx2char"], data["char2idx"].item())

    return tokenizer, train_loader, val_loader

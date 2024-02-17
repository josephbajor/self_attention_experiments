import numpy as np
import pickle
import torch
import requests
import torch.nn.functional as F

import os
import glob
from pathlib import Path

from src import logger

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

import sys


def preprocess_wikitext_bpe(path):
    """
    Preprocess wikitext by using bytepair encoding
    """

    files = glob.glob(f"{path}/*.tokens")

    # ensure that there is data in the pointed directory
    if len(files) == 0:
        raise AssertionError(
            f"no token files found in {path} (is you data directory correct?)"
        )

    tokenizer = Tokenizer(BPE())

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(special_tokens=["<unk>"])
    tokenizer.train(files=files, trainer=trainer)

    data_out = {}
    for file in files:
        fname = file.split("/")[-1]
        with open(file, "r") as f:
            lines = f.readlines()
        tokens = tokenizer.encode_batch(lines)
        data_out[fname] = np.concatenate(
            [np.array(t.ids, dtype=np.int32) for t in tokens]
        )

    return data_out, tokenizer


def preprocess_wikitext_wordpeice(path, vocab_size):
    files = glob.glob(f"{path}/*.tokens")

    # ensure that there is data in the pointed directory
    if len(files) == 0:
        raise AssertionError(
            f"no token files found in {path} (is you data directory correct?)"
        )

    tokenizer = Tokenizer(WordPiece())

    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=["<unk>", "@-@"], show_progress=True
    )
    tokenizer.train(files=files, trainer=trainer)

    data_out = {}
    for file in files:
        fname = file.split("/")[-1]
        with open(file, "r") as f:
            lines = f.readlines()
        tokens = tokenizer.encode_batch(lines)
        data_out[fname] = np.concatenate(
            [np.array(t.ids, dtype=np.int32) for t in tokens]
        )

    return data_out, tokenizer


def preprocess_wikitext_char(path):
    """
    Preprocess wikitext using character tokenization
    """
    return NotImplementedError


def preprocess_shakespeare_char(path: Path, split: float = 0.8) -> None:
    """
    Download and preprocess the shakespeare dataset for character level tokenization

    Args:
        path (Path): the path to save the data to
        split (float, optional): the fraction of the data to use for training. Defaults to 0.8.
    """

    if not isinstance(path, Path):
        path = Path(path)

    data_path = path / "shakespeare.txt"

    if not os.path.exists(data_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(data_path, "w") as f:
            f.write(requests.get(data_url).text)

    with open(data_path, "r") as f:
        data = f.read()

    # get the unique characters in the text
    vocab = sorted(set(data))

    # create a mapping from characters to indices and vice versa
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    # tokenize at the character level
    data = np.array([char2idx[c] for c in data])

    # create a training and validation split
    train_data = data[: int(split * len(data))]
    val_data = data[int(split * len(data)) :]

    # save the data to disk
    np.savez_compressed(
        path / "shakespeare.npz",
        train=train_data,
        val=val_data,
        idx2char=idx2char,
        char2idx=char2idx,
    )


if __name__ == "__main__":
    # Preprocess and save the data to disk
    # this avoids the issues with huggingface tokenizer parrelelism
    # also speeds things up quite a bit

    # we take command line arguments to indicate whether we want to set up wikitext or shakespeare
    import argparse
    from hparams import Hparams

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="wikitext", help="The dataset to preprocess"
    )
    args = parser.parse_args()

    if args.dataset == "wikitext":
        logger.info("Tokenizing data...")
        hparams = Hparams()
        os.makedirs(hparams.tokenized_dir, exist_ok=True)
        data, tokenizer = preprocess_wikitext_wordpeice(
            hparams.path, hparams.vocab_size
        )

        logger.info("Saving tokenized data...")
        np.savez_compressed(
            f"{hparams.tokenized_dir}/wikitext_tokenized.npz",
            train=data["wiki.train.tokens"],
            val=data["wiki.valid.tokens"],
            test=data["wiki.test.tokens"],
        )

        logger.info("Saving tokenizer...")
        tokenizer.save(f"{hparams.tokenized_dir}/tokenizer.json")

    if args.dataset == "shakespeare":
        hparams = Hparams()
        preprocess_shakespeare_char(hparams.tokenized_dir, hparams.train_split)
        logger.info("Shakespeare data preprocessed and saved to disk")

    else:
        raise ValueError(f"Invalid dataset argument {args.dataset}")

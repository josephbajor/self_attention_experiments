import numpy as np
import pickle
import torch
import torch.nn.functional as F

import os
import glob

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

    #ensure that there is data in the pointed directory
    if len(files) == 0:
        raise AssertionError(f'no token files found in {path} (is you data directory correct?)')

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

    #ensure that there is data in the pointed directory
    if len(files) == 0:
        raise AssertionError(f'no token files found in {path} (is you data directory correct?)')

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


if __name__ == "__main__":
    # Preprocess and save the data to disk
    # this avoids the issues with huggingface tokenizer parrelelism
    # also speeds things up quite a bit

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

    from hparams import Hparams

    print("Tokenizing data...")
    hparams = Hparams()
    os.makedirs(hparams.tokenized_dir, exist_ok=True)
    data, tokenizer = preprocess_wikitext_wordpeice(hparams.path, hparams.vocab_size)

    print("Saving tokenized data...")
    np.savez_compressed(
        f"{hparams.tokenized_dir}/wikitext_tokenized.npz",
        train=data["wiki.train.tokens"],
        val=data["wiki.valid.tokens"],
        test=data["wiki.test.tokens"],
    )

    print("Saving tokenizer...")
    tokenizer.save(f"{hparams.tokenized_dir}/tokenizer.json")

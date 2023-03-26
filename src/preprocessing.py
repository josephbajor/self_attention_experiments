import numpy as np
import pickle
import torch
import torch.nn.functional as F

import os
import glob

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from src.model import SimpleBigramModel


def preprocess_wikitext(path):
    """
    Preprocess wikitext by using bytepair encoding
    """

    files = glob.glob(f'{path}/*.tokens')

    tokenizer = Tokenizer(BPE())

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(special_tokens=['<unk>'])
    tokenizer.train(files = files, trainer = trainer)

    data_out = {}
    for file in files:
        fname = file.split('/')[-1]
        with open(file, 'r') as f:
            lines = f.readlines()
        tokens = tokenizer.encode_batch(lines)
        data_out[fname] = np.concatenate([np.array(t.ids, dtype=np.int32) for t in tokens])

    return data_out

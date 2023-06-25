from dataclasses import dataclass


@dataclass
class Hparams:
    path: str = (
        "/home/jbajor/Dev/CMU/Research/fairseq/examples/language_model/wikitext-103"
    )
    tokenized_dir: str = "data"
    vocab_size: int = 3000

    num_workers = 4
    batch_size = 128
    epochs = 25

    max_span = 64
    embed_size = 128
    att_block_size = 64
    num_heads = 3

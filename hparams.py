from dataclasses import dataclass


@dataclass
class Hparams:
    path: str = (
        "/home/jbajor/Dev/CMU/Research/fairseq/examples/language_model/wikitext-103"
    )
    tokenized_dir: str = "data"

    num_workers = 4
    batch_size = 64

    block_size = 128

from dataclasses import dataclass


@dataclass
class Hparams:
    path: str = "wikitext-103"
    tokenized_dir: str = "data"
    vocab_size: int = 3000

    num_workers = 4
    batch_size = 128
    epochs = 5

    max_span = 64
    embed_size = 128
    att_block_size = 256
    num_heads = 2

    use_positional_embedding: bool = True

    windowed_loss_buffer_size: int = 100

    universal_pos_enc: bool = False

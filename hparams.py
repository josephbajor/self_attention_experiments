from dataclasses import dataclass, asdict
import json
from typing import Literal, Optional
from pathlib import Path


@dataclass
class Hparams:
    ### data ###
    path: str = "wikitext-103"
    tokenized_dir: str = "data"
    vocab_size: int = 2000
    model_directory: str = "checkpoints"
    logging_path: str = "logs"
    random_seed: int = 42
    train_split: float = 0.8  # only used for shakespeare currently
    dataset: Literal["wikitext", "shakespeare"] = "shakespeare"

    ### dataloaders ###
    num_workers: int = 6

    ### training ###
    batch_size: int = 64
    epochs: int = 2
    windowed_loss_buffer_size: int = 60
    compile_model = True
    eval_steps: Optional[int] = 20  # set to none for whole val dataset
    eval_every_n_steps: int = 30

    ### model ###
    type: Literal["gpt", "nanogpt"] = "nanogpt"

    max_span: int = 256
    embed_size: int = 384
    # att_block_size: int = 256
    num_heads: int = 6
    att_layers: int = 6
    ff_internal_mult: int = 4
    dropout: float = 0.2

    att_func_type: Literal["full", "fnet"] = "full"
    emb_func: Optional[Literal["binary_static", "binary_learned"]] = None
    use_positional_embedding: bool = False
    use_flash: bool = False

    universal_pos_enc: bool = False

    ### optimizer ###
    lr = 1e-3
    weight_decay = 0.01
    gradient_clip_val = 1.0
    half_precision = False

    ### logging ###

    def save_to_file(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(json.dumps(asdict(self), indent=4))

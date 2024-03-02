from dataclasses import dataclass, asdict
import json
from typing import Literal, Optional
from pathlib import Path


@dataclass
class Hparams:
    ### data ###
    path: str = "wikitext-103"
    tokenized_dir: str = "data"
    vocab_size: int = 2250
    model_directory: str = "checkpoints"
    logging_path: str = "logs"
    random_seed: int = 42
    train_split: float = 0.8  # only used for shakespeare currently
    dataset: Literal["wikitext", "shakespeare"] = "wikitext"

    ### dataloaders ###
    num_workers: int = 6

    ### training ###
    batch_size: int = 64
    epochs: int = 2
    windowed_loss_buffer_size: int = 60
    compile_model = True
    eval_steps: Optional[int] = 25  # set to none for whole val dataset
    eval_every_n_steps: int = 400

    ### model ###
    type: Literal["gpt", "nanogpt"] = "gpt"

    max_span: int = 256
    embed_size: int = 300
    # att_block_size: int = 256
    num_heads: int = 6
    att_layers: int = 8
    ff_internal_mult: int = 3
    dropout: float = 0.2

    att_func_type: Literal["full", "fnet"] = "full"
    emb_func: Optional[Literal["binary_static", "binary_learned"]] = "binary_learned"
    use_positional_embedding: bool = False
    use_flash: bool = True

    universal_pos_enc: bool = False

    ### optimizer ###
    lr = 1e-3
    weight_decay = 0.01
    gradient_clip_val = 1.0
    half_precision = True

    ### logging ###

    def save_to_file(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(json.dumps(asdict(self), indent=4))

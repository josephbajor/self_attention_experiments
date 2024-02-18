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
    windowed_loss_buffer_size: int = 100
    compile_model = True
    eval_steps: Optional[int] = 30  # set to none for whole val dataset

    ### model ###
    max_span: int = 256
    embed_size: int = 512
    # att_block_size: int = 256
    num_heads: int = 3
    att_layers: int = 4
    # ff_hidden_size: int = 256
    dropout: float = 0.08

    att_func_type: Literal["full", "fnet"] = "full"
    emb_func: Optional[Literal["binary_static", "binary_learned"]] = None
    use_positional_embedding: bool = True
    use_flash: bool = True

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

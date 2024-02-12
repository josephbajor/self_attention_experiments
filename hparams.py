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

    ### dataloaders ###
    num_workers: int = 6

    ### training ###
    batch_size: int = 128
    epochs: int = 2
    windowed_loss_buffer_size: int = 100
    compile_model = False
    eval_steps: Optional[int] = 80  # set to none for whole val dataset

    ### model ###
    max_span: int = 128
    embed_size: int = 512
    att_block_size: int = 256
    num_heads: int = 3
    att_layers: int = 4
    ff_hidden_size: int = 256
    dropout: float = 0.08

    att_func_type: Literal["full", "fnet"] = "full"
    emb_func: Optional[Literal["binary_static", "binary_learned"]] = None
    use_positional_embedding: bool = True
    use_flash: bool = True

    universal_pos_enc: bool = False

    ### optimizer ###
    lr = 0.002
    weight_decay = 0.01

    ### logging ###

    def save_to_file(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(json.dumps(asdict(self), indent=4))

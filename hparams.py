from dataclasses import dataclass, asdict
import json


@dataclass
class Hparams:
    ### data ###
    path: str = "wikitext-103"
    tokenized_dir: str = "data"
    vocab_size: int = 3000
    model_directory: str = "checkpoints"

    ### dataloaders ###
    num_workers = 4

    ### training ###
    batch_size = 128
    epochs = 5

    ### model ###
    max_span = 64
    embed_size = 128
    att_block_size = 256
    num_heads = 3
    att_layers = 4
    ff_hidden_size = 256
    dropout = 0.1

    use_positional_embedding: bool = True

    universal_pos_enc: bool = False

    ### optimizer ###
    lr = 0.001
    weight_decay = 0.0

    ### logging ###

    def save_to_file(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(json.dumps(asdict(self), indent=4))

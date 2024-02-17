import torch
from src.model.LM import AttentionLM
from typing import Literal


def inference(
    input: str,
    model: AttentionLM,
    tokenizer,
    out_len: int,
    mode: Literal["wikitext", "shakespeare"] = "wikitext",
    determenistic=False,
    device="cuda",
):
    model.eval()
    with torch.inference_mode():
        if mode == "wikitext":
            input = torch.tensor(tokenizer.encode(input).ids, dtype=torch.long).to(
                device
            )
            out = model.generate_batch(input, out_len, deterministic=determenistic)
            out = tokenizer.decode(list(out.squeeze().detach().cpu()))
        if mode == "shakespeare":
            input = torch.tensor(tokenizer.encode(input), dtype=torch.long).to(device)
            out = model.generate_batch(input, out_len, deterministic=determenistic)
            out = tokenizer.decode(list(out.squeeze().detach().cpu()))
        return out

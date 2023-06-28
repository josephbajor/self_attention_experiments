import torch

def inference(input:str, model, tokenizer, out_len:int, determenistic=False, device = 'cuda'):
    model.eval()
    with torch.inference_mode():
        input = torch.tensor(tokenizer.encode(input).ids, dtype=torch.long).to(device)
        out = model.generate_batch(input, out_len, deterministic=determenistic)
        out = [tokenizer.decode(list(t)) for t in out]
        return out
    
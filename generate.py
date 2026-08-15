import argparse

import tiktoken
import torch

import AttnRes

import os

os.system('clear')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = AttnRes.GPT.LanguageModel(
        **checkpoint["model_kwargs"]
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["block_size"]


@torch.no_grad()
def generate(model, block_size, idx, max_new_tokens, temperature, top_k):
    # idx: (1, S) tensor of token ids, grown one token at a time
    context_len = block_size - 1
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_len:]
        logits = model(idx_cond)[...,-1,:]

        if temperature <= 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()
    enc = tiktoken.get_encoding("gpt2")
    model, block_size = load_model(args.checkpoint)

    ids = enc.encode_ordinary(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    out = generate(
        model, block_size, idx, args.max_new_tokens, args.temperature, args.top_k
    )
    print(f"Example token: {enc.decode([1000])}\n")
    print(enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()

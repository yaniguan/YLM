"""Run a frozen text encoder once on every unique question and save the token states.

The text encoder is frozen in month 1, so there is no reason to run it during training.

Output: data/text_<name>.pt = {"dim": d, "states": {question: tensor (T, d)}}

Run:
  python scripts/03_embed_text.py --name minilm
  python scripts/03_embed_text.py --name matscibert
"""
import argparse
import json

import torch
from transformers import AutoModel, AutoTokenizer

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "matscibert": "m3rg-iitd/matscibert",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=list(MODELS), required=True)
    parser.add_argument("--questions", default="data/tensorqa_v0.json")
    args = parser.parse_args()

    questions = sorted({q["question"] for q in json.load(open(args.questions))})
    tok = AutoTokenizer.from_pretrained(MODELS[args.name])
    model = AutoModel.from_pretrained(MODELS[args.name]).eval()

    states = {}
    with torch.no_grad():
        for q in questions:
            enc = tok(q, return_tensors="pt")
            states[q] = model(**enc).last_hidden_state[0].clone()   # (T, d), includes [CLS] and [SEP]
    dim = next(iter(states.values())).shape[-1]
    torch.save({"dim": dim, "states": states}, f"data/text_{args.name}.pt")
    print(f"{args.name}: {len(states)} questions, dim {dim}, saved data/text_{args.name}.pt")


if __name__ == "__main__":
    main()

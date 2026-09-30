"""Baseline (ii): write the coordinates as text and let a small LLM answer.

Prompt:
    Atom positions in bohr:
    O 1.23 -0.45 0.67
    C ...
    Question: What is the force on the oxygen atom bonded to two carbons?
    Answer:

The LLM (Qwen2.5, fully fine-tuned) reads the prompt. The hidden state of the last token goes into a
Linear layer -> 3 numbers. (A regression head, not number tokens, so the comparison is about
*reading* geometry, not about how numbers are tokenized.)

--augment : rotate every training molecule randomly (the usual trick to "teach" LLMs symmetry)

Evaluation: same rotation protocol as scripts/04_train.py, plus
    mae_orig_frame : error when the molecule is shown in the dataset's own orientation (no rotation)
The gap between mae_orig_frame and mae shows how much the model depends on that orientation.

Run:
  python scripts/05_train_llm_baseline.py --llm Qwen/Qwen2.5-0.5B --name llm05 [--augment]
"""
import argparse
import json
import math
import os
import pickle
import time

import numpy as np
import torch
from e3nn import o3
from transformers import AutoModel, AutoTokenizer

TASKS = {"dipole": 0, "force": 1}
SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def make_prompt(numbers, pos, question):
    lines = ["Atom positions in bohr:"]
    for z, (x, y, w) in zip(numbers, pos):
        lines.append(f"{SYMBOL[int(z)]} {x:.2f} {y:.2f} {w:.2f}")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


class LLMRegressor(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.llm = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16)
        self.head = torch.nn.Linear(self.llm.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        h = self.llm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(1) - 1                       # right padding: last real token
        h_last = h[torch.arange(h.shape[0]), last].float()
        return self.head(h_last)


def load(labels_path, questions_path):
    labels = pickle.load(open(labels_path, "rb"))
    questions = json.load(open(questions_path))
    splits = {s: [q for q in questions if q["split"] == s] for s in ["train", "val", "test", "test_heldout"]}
    return labels, splits


def get_batch(labels, qs, tok, device, R=None, random_rotation=False):
    prompts, targets, tasks = [], [], []
    for q in qs:
        lab = labels[q["mol"]]
        pos = torch.tensor(lab["pos"], dtype=torch.float32)
        target = torch.tensor(lab["dipole"] if q["task"] == "dipole" else lab["forces"][q["atom"]],
                              dtype=torch.float32)
        rot = o3.rand_matrix() if random_rotation else R
        if rot is not None:
            pos, target = pos @ rot.T, target @ rot.T
        prompts.append(make_prompt(lab["numbers"], pos.tolist(), q["question"]))
        targets.append(target)
        tasks.append(TASKS[q["task"]])
    enc = tok(prompts, return_tensors="pt", padding=True)
    return (enc["input_ids"].to(device), enc["attention_mask"].to(device),
            torch.stack(targets).to(device), torch.tensor(tasks, device=device))


@torch.no_grad()
def evaluate(model, labels, qs, tok, scale, device, n_rot, batch_size=64, seed=0):
    model.eval()
    rotations = [torch.eye(3)]
    if n_rot > 1:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            rotations += list(o3.rand_matrix(n_rot - 1))
    res = {t: {"abs_err": [], "abs_err_orig": [], "cos": [], "eq_err": []} for t in TASKS}
    for start in range(0, len(qs), batch_size):
        chunk = qs[start: start + batch_size]
        y0 = None
        for r, R in enumerate(rotations):
            ids, am, target, task = get_batch(labels, chunk, tok, device, R=R)
            y = (model(ids, am) * scale[task][:, None]).cpu()
            target, task = target.cpu(), task.cpu()
            err = (y - target).abs().mean(-1)
            cos = torch.nn.functional.cosine_similarity(y, target, dim=-1)
            if y0 is None:
                y0 = y
            else:
                eq = (y - y0 @ R.T).norm(dim=-1) / y0.norm(dim=-1).clamp(min=1e-12)
            for name, k in TASKS.items():
                sel = task == k
                res[name]["abs_err"] += err[sel].tolist()
                res[name]["cos"] += cos[sel].tolist()
                if r == 0:
                    res[name]["abs_err_orig"] += err[sel].tolist()
                else:
                    res[name]["eq_err"] += eq[sel].tolist()
    model.train()
    return {name: {"mae": float(np.mean(r["abs_err"])),
                   "mae_orig_frame": float(np.mean(r["abs_err_orig"])),
                   "cosine": float(np.mean(r["cos"])),
                   "eq_err_mean": float(np.mean(r["eq_err"])) if r["eq_err"] else None,
                   "eq_err_max": float(np.max(r["eq_err"])) if r["eq_err"] else None}
            for name, r in res.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--name", required=True)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--eval_rotations", type=int, default=8)
    p.add_argument("--val_size", type=int, default=2000, help="val questions checked each epoch")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    out_dir = f"runs/{args.name}"
    os.makedirs(out_dir, exist_ok=True)
    log = open(f"{out_dir}/log.txt", "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    labels, splits = load("data/qm9_xtb.pkl", "data/tensorqa_v0.json")
    train = splits["train"]
    scale = torch.tensor([
        math.sqrt(np.mean([np.mean(np.square(labels[q["mol"]]["dipole"])) for q in train if q["task"] == "dipole"])),
        math.sqrt(np.mean([np.mean(np.square(labels[q["mol"]]["forces"][q["atom"]])) for q in train if q["task"] == "force"])),
    ], device=device)
    say("train questions", len(train), "scale", scale.tolist(), "augment", args.augment)

    tok = AutoTokenizer.from_pretrained(args.llm)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = LLMRegressor(args.llm).to(device)
    opt = torch.optim.AdamW([
        {"params": model.llm.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=0.0)
    steps = args.epochs * math.ceil(len(train) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, 1e-3], total_steps=steps, pct_start=0.05)

    rng = np.random.default_rng(args.seed)
    val_qs = [splits["val"][i] for i in rng.choice(len(splits["val"]), args.val_size, replace=False)]
    best, t0 = float("inf"), time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(len(train))
        losses = []
        for start in range(0, len(order), args.batch_size):
            chunk = [train[i] for i in order[start: start + args.batch_size]]
            ids, am, target, task = get_batch(labels, chunk, tok, device, random_rotation=args.augment)
            y = model(ids, am)
            loss = ((y - target / scale[task][:, None]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            if len(losses) % 200 == 0:
                say(f"epoch {epoch} step {len(losses)} loss {np.mean(losses[-200:]):.4f} {time.time() - t0:.0f}s")
        val = evaluate(model, labels, val_qs, tok, scale, device, n_rot=1)
        score = val["dipole"]["mae"] / scale[0].item() + val["force"]["mae"] / scale[1].item()
        say(f"epoch {epoch} val {json.dumps(val)}")
        if score < best:
            best = score
            torch.save(model.state_dict(), f"{out_dir}/model.pt")

    model.load_state_dict(torch.load(f"{out_dir}/model.pt"))
    results = {"args": vars(args), "scale": scale.tolist()}
    for split in ["test", "test_heldout"]:
        results[split] = evaluate(model, labels, splits[split], tok, scale, device, n_rot=args.eval_rotations)
        say(split, json.dumps(results[split]))
    json.dump(results, open(f"{out_dir}/results.json", "w"), indent=2)
    os.remove(f"{out_dir}/model.pt")  # 1-3 GB each; scratch space is not free
    say("saved", out_dir)


if __name__ == "__main__":
    main()

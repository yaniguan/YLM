"""Train and evaluate the month-1 model on TensorQA-v0 (dipole + force questions).

Run (examples):
  python scripts/04_train.py --text minilm     --name ylm_minilm
  python scripts/04_train.py --text matscibert --name ylm_matscibert
  python scripts/04_train.py --text taskid     --name base_taskid      # baseline v

Output: runs/<name>/results.json, runs/<name>/model.pt, runs/<name>/log.txt

Evaluation (rotation protocol, 04_data_and_benchmark.md):
  every test question is run under K random rotations. We report
    mae       : mean absolute error per vector component, physical units (e*bohr or hartree/bohr)
    cosine    : cosine between predicted and true vector (direction correct?)
    eq_err    : || f(R x) - R f(x) || / || f(x) ||   (should be ~1e-6 in float32)
    atom_acc  : force questions only. Does the pooled attention peak on the asked atom?
"""
import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
from e3nn import o3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import YLM  # noqa: E402

TASKS = {"dipole": 0, "force": 1}
N_MAX = 29  # largest QM9 molecule


def load_data(labels_path, questions_path, text_path):
    labels = pickle.load(open(labels_path, "rb"))
    questions = json.load(open(questions_path))

    # molecules as padded tensors
    M = len(labels)
    Z = torch.zeros(M, N_MAX, dtype=torch.long)
    pos = torch.zeros(M, N_MAX, 3)
    mask = torch.zeros(M, N_MAX, dtype=torch.bool)
    for m, lab in enumerate(labels):
        n = len(lab["numbers"])
        Z[m, :n] = torch.tensor(lab["numbers"])
        pos[m, :n] = torch.tensor(lab["pos"], dtype=torch.float32)
        mask[m, :n] = True

    # text: one padded tensor per unique question
    if text_path is None:
        text_dim, T = 16, 1
        q_text = {q["question"]: 0 for q in questions}
        text = torch.zeros(1, T, text_dim)
        text_mask = torch.ones(1, T, dtype=torch.bool)
    else:
        cache = torch.load(text_path)
        text_dim = cache["dim"]
        strings = sorted(cache["states"])
        T = max(s.shape[0] for s in cache["states"].values())
        text = torch.zeros(len(strings), T, text_dim)
        text_mask = torch.zeros(len(strings), T, dtype=torch.bool)
        for k, s in enumerate(strings):
            st = cache["states"][s]
            text[k, : st.shape[0]] = st
            text_mask[k, : st.shape[0]] = True
        q_text = {s: k for k, s in enumerate(strings)}

    # questions
    splits = {}
    for split in ["train", "val", "test", "test_heldout"]:
        qs = [q for q in questions if q["split"] == split]
        target = []
        for q in qs:
            lab = labels[q["mol"]]
            target.append(lab["dipole"] if q["task"] == "dipole" else lab["forces"][q["atom"]])
        splits[split] = {
            "mol": torch.tensor([q["mol"] for q in qs]),
            "text": torch.tensor([q_text[q["question"]] for q in qs]),
            "task": torch.tensor([TASKS[q["task"]] for q in qs]),
            "atom": torch.tensor([q["atom"] for q in qs]),
            "target": torch.tensor(np.array(target), dtype=torch.float32),
        }
    return dict(Z=Z, pos=pos, mask=mask, text=text, text_mask=text_mask, text_dim=text_dim), splits


def get_batch(data, split, idx, device, R=None):
    m = split["mol"][idx]
    pos = data["pos"][m]
    target = split["target"][idx]
    if R is not None:
        pos = pos @ R.T
        target = target @ R.T
    t = split["text"][idx]
    return (data["Z"][m].to(device), pos.to(device), data["mask"][m].to(device),
            data["text"][t].to(device), data["text_mask"][t].to(device),
            split["task"][idx].to(device)), target.to(device)


@torch.no_grad()
def evaluate(model, data, split, scale, device, n_rot, batch_size=512, seed=0):
    model.eval()
    rotations = [torch.eye(3)]  # first one = no rotation
    if n_rot > 1:
        with torch.random.fork_rng():  # same rotations every time, without touching the training RNG
            torch.manual_seed(seed)
            rotations += list(o3.rand_matrix(n_rot - 1))
    n = len(split["mol"])
    res = {t: {"abs_err": [], "cos": [], "eq_err": [], "atom_hit": []} for t in TASKS}
    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n))
        task = split["task"][idx]
        y0 = None
        for R in rotations:
            inputs, target = get_batch(data, split, idx, device, R)
            y, attn = model(*inputs)
            y = (y * scale[inputs[5]][:, None]).cpu()
            target = target.cpu()
            if y0 is None:
                y0 = y  # prediction in the original frame (R = identity)
            else:
                eq = (y - y0 @ R.T).norm(dim=-1) / y0.norm(dim=-1).clamp(min=1e-12)
                for name, k in TASKS.items():
                    res[name]["eq_err"] += eq[task == k].tolist()
            cos = torch.nn.functional.cosine_similarity(y, target, dim=-1)
            for name, k in TASKS.items():
                sel = task == k
                res[name]["abs_err"] += (y - target)[sel].abs().mean(-1).tolist()
                res[name]["cos"] += cos[sel].tolist()
            hit = (attn.argmax(-1).cpu() == split["atom"][idx])
            res["force"]["atom_hit"] += hit[task == 1].tolist()
    out = {}
    for name, r in res.items():
        out[name] = {
            "mae": float(np.mean(r["abs_err"])),
            "cosine": float(np.mean(r["cos"])),
            "eq_err_mean": float(np.mean(r["eq_err"])) if r["eq_err"] else None,
            "eq_err_max": float(np.max(r["eq_err"])) if r["eq_err"] else None,
        }
        if name == "force":
            out[name]["atom_acc"] = float(np.mean(r["atom_hit"]))
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", choices=["minilm", "matscibert", "taskid"], required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train_fraction", type=float, default=1.0, help="for sample-efficiency curves")
    p.add_argument("--eval_rotations", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=0, help="debug: stop early")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = f"runs/{args.name}"
    os.makedirs(out_dir, exist_ok=True)
    log = open(f"{out_dir}/log.txt", "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    text_path = None if args.text == "taskid" else f"data/text_{args.text}.pt"
    data, splits = load_data("data/qm9_xtb.pkl", "data/tensorqa_v0.json", text_path)
    train = splits["train"]
    if args.train_fraction < 1.0:
        keep = torch.randperm(len(train["mol"]))[: int(args.train_fraction * len(train["mol"]))]
        train = {k: v[keep] for k, v in train.items()}

    # one scale per task (RMS of the train targets), so both tasks have loss ~1
    scale = torch.stack([train["target"][train["task"] == k].pow(2).mean().sqrt() for k in TASKS.values()])
    say("device", device, "| train questions", len(train["mol"]), "| scale", scale.tolist())
    scale_dev = scale.to(device)

    model = YLM(text_dim=data["text_dim"], text_mode="taskid" if args.text == "taskid" else "text").to(device)
    say("parameters", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps_per_epoch = math.ceil(len(train["mol"]) / args.batch_size)
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)

    step, best_val, t0 = 0, float("inf"), time.time()
    for epoch in range(args.epochs):
        perm = torch.randperm(len(train["mol"]))
        losses = []
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start: start + args.batch_size]
            inputs, target = get_batch(data, train, idx, device)
            y, _ = model(*inputs)
            loss = ((y - target / scale_dev[inputs[5]][:, None]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        val = evaluate(model, data, splits["val"], scale, device, n_rot=1)
        val_score = val["dipole"]["mae"] / scale[0].item() + val["force"]["mae"] / scale[1].item()
        if val_score < best_val:
            best_val = val_score
            torch.save(model.state_dict(), f"{out_dir}/model.pt")
        say(f"epoch {epoch} loss {np.mean(losses):.4f} | val dipole mae {val['dipole']['mae']:.4f} "
            f"cos {val['dipole']['cosine']:.3f} | force mae {val['force']['mae']:.5f} "
            f"cos {val['force']['cosine']:.3f} atom_acc {val['force']['atom_acc']:.3f} | {time.time() - t0:.0f}s")
        if args.max_steps and step >= args.max_steps:
            break

    model.load_state_dict(torch.load(f"{out_dir}/model.pt"))
    results = {"args": vars(args), "scale": scale.tolist()}
    for split in ["test", "test_heldout"]:
        results[split] = evaluate(model, data, splits[split], scale, device, n_rot=args.eval_rotations)
        say(split, json.dumps(results[split]))
    json.dump(results, open(f"{out_dir}/results.json", "w"), indent=2)
    say("saved", out_dir)


if __name__ == "__main__":
    main()

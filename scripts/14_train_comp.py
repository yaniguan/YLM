"""Train / evaluate on the compositional questions (T4 diff, T5 proj) of data/tensorqa_comp.json.

  python scripts/14_train_comp.py --model bc   --name comp_bc_s0      # Option B/C-lite (src/option_bc.py)
  python scripts/14_train_comp.py --model optA --name comp_optA_s0    # Option A only (single pointer)

Same rotation protocol as scripts/04_train.py. Needs data/text_comp_<text>.pt (03_embed_text.py --out ...).
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
from src.option_bc import CompModel  # noqa: E402

TASKS = {"diff": 0, "proj": 1}
N_MAX = 29


def load(labels_path, qpath, tpath):
    labels = pickle.load(open(labels_path, "rb"))
    qs = json.load(open(qpath))
    cache = torch.load(tpath)
    strings = sorted(cache["states"])
    T = max(s.shape[0] for s in cache["states"].values())
    text = torch.zeros(len(strings), T, cache["dim"])
    tmask = torch.zeros(len(strings), T, dtype=torch.bool)
    for k, s in enumerate(strings):
        st = cache["states"][s]
        text[k, : st.shape[0]] = st
        tmask[k, : st.shape[0]] = True
    sid = {s: k for k, s in enumerate(strings)}
    M = len(labels)
    Z = torch.zeros(M, N_MAX, dtype=torch.long)
    pos = torch.zeros(M, N_MAX, 3)
    mask = torch.zeros(M, N_MAX, dtype=torch.bool)
    for m, lab in enumerate(labels):
        n = len(lab["numbers"])
        Z[m, :n] = torch.tensor(lab["numbers"])
        pos[m, :n] = torch.tensor(lab["pos"], dtype=torch.float32)
        mask[m, :n] = True
    splits = {}
    for sp in ["train", "val", "test", "test_heldout"]:
        sel = [q for q in qs if q["split"] == sp]
        splits[sp] = dict(mol=torch.tensor([q["mol"] for q in sel]),
                          text=torch.tensor([sid[q["question"]] for q in sel]),
                          task=torch.tensor([TASKS[q["task"]] for q in sel]),
                          atoms=torch.tensor([q["atoms"] for q in sel]),
                          target=torch.tensor([q["target"] for q in sel], dtype=torch.float32))
    return dict(Z=Z, pos=pos, mask=mask, text=text, tmask=tmask, dim=cache["dim"]), splits


def batch(data, sp, idx, device, R=None):
    m = sp["mol"][idx]
    pos, tgt = data["pos"][m], sp["target"][idx]
    if R is not None:
        pos, tgt = pos @ R.T, tgt @ R.T
    t = sp["text"][idx]
    return (data["Z"][m].to(device), pos.to(device), data["mask"][m].to(device),
            data["text"][t].to(device), data["tmask"][t].to(device)), tgt.to(device), sp["task"][idx].to(device)


def forward(model, kind, inputs, task):
    if kind == "optA":  # YLM takes a task id only in "taskid" mode; here it reads the text
        return model(*inputs, task)[0]
    return model(*inputs)[0]


@torch.no_grad()
def evaluate(model, kind, data, sp, scale, device, n_rot, bs=512):
    model.eval()
    scale = scale.to(device)
    rots = [torch.eye(3)]
    if n_rot > 1:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            rots += list(o3.rand_matrix(n_rot - 1))
    res = {t: dict(ae=[], cos=[], eq=[]) for t in TASKS}
    for s in range(0, len(sp["mol"]), bs):
        idx = torch.arange(s, min(s + bs, len(sp["mol"])))
        y0 = None
        for R in rots:
            inp, tgt, task = batch(data, sp, idx, device, R)
            y = (forward(model, kind, inp, task) * scale[task][:, None]).cpu()
            tgt, task = tgt.cpu(), task.cpu()
            if y0 is None:
                y0 = y
            else:
                eq = (y - y0 @ R.T).norm(dim=-1) / y0.norm(dim=-1).clamp(min=1e-12)
            cos = torch.nn.functional.cosine_similarity(y, tgt, dim=-1)
            for n, k in TASKS.items():
                sel = task == k
                res[n]["ae"] += (y - tgt).abs().mean(-1)[sel].tolist()
                res[n]["cos"] += cos[sel].tolist()
                if y0 is not y:
                    res[n]["eq"] += eq[sel].tolist()
    model.train()
    return {n: dict(mae=float(np.mean(r["ae"])), cosine=float(np.mean(r["cos"])),
                    eq_err_mean=float(np.mean(r["eq"])) if r["eq"] else None) for n, r in res.items()}


@torch.no_grad()
def diagnose(model, data, sp, device, bs=512):
    """Where do the two pointers look, and which coefficients does the model choose?

    hit[h][j] = fraction of questions where head h's argmax atom is the question's atom j (0 = A, 1 = B).
    A model that follows the wording has hit[0] = (1, 0), hit[1] = (0, 1) for "A ... B" order and the
    reverse for "Subtract B from A"; coef = mean (c1..c4) per task; ideal diff (1,-1,0,0), proj (0,0,1,0).
    """
    model.eval()
    hits = {t: torch.zeros(2, 2) for t in TASKS}
    coef = {t: torch.zeros(4) for t in TASKS}
    n = {t: 0 for t in TASKS}
    for s in range(0, len(sp["mol"]), bs):
        idx = torch.arange(s, min(s + bs, len(sp["mol"])))
        inp, _, task = batch(data, sp, idx, device)
        model(*inp)
        arg = [a.argmax(-1).cpu() for a in model.last["attn"]]
        atoms, task, c = sp["atoms"][idx], task.cpu(), model.last["c"].cpu()
        for t, k in TASKS.items():
            sel = task == k
            n[t] += int(sel.sum())
            coef[t] += c[sel].sum(0)
            for h in range(2):
                for j in range(2):
                    hits[t][h, j] += (arg[h][sel] == atoms[sel][:, j]).sum()
    model.train()
    return {t: dict(hit=(hits[t] / n[t]).tolist(), coef=(coef[t] / n[t]).tolist()) for t in TASKS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["bc", "optA"], required=True)
    p.add_argument("--text", default="matscibert")
    p.add_argument("--data", default="tensorqa_comp", help="questions file data/<data>.json; text cache data/text_<data>_<text>.pt")
    p.add_argument("--token_dropout", type=float, default=0.0, help="bc only: drop this fraction of text tokens in training")
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval_rotations", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--test_limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = f"runs/{args.name}"
    os.makedirs(out, exist_ok=True)
    log = open(f"{out}/log.txt", "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    data, splits = load("data/qm9_xtb.pkl", f"data/{args.data}.json", f"data/text_{args.data.replace('tensorqa_', '')}_{args.text}.pt")
    train = splits["train"]
    scale = torch.stack([train["target"][train["task"] == k].pow(2).mean().sqrt() for k in TASKS.values()])
    sd = scale.to(device)
    model = (CompModel(data["dim"], token_dropout=args.token_dropout) if args.model == "bc" else YLM(text_dim=data["dim"])).to(device)
    say("device", device, "| params", sum(q.numel() for q in model.parameters()), "| scale", scale.tolist())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total = args.epochs * math.ceil(len(train["mol"]) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    best, step, t0 = float("inf"), 0, time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(len(train["mol"]))
        ls = []
        for s in range(0, len(perm), args.batch_size):
            inp, tgt, task = batch(data, train, perm[s: s + args.batch_size], device)
            loss = ((forward(model, args.model, inp, task) - tgt / sd[task][:, None]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            ls.append(loss.item())
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        val = evaluate(model, args.model, data, splits["val"], scale, device, 1)
        score = sum(val[t]["mae"] / scale[k].item() for t, k in TASKS.items())
        if score < best:
            best = score
            torch.save(model.state_dict(), f"{out}/model.pt")
        say(f"epoch {ep} loss {np.mean(ls):.4f} | " + " | ".join(f"{t} mae {val[t]['mae']:.4f} cos {val[t]['cosine']:.3f}" for t in TASKS)
            + f" | {time.time() - t0:.0f}s")
        if args.max_steps and step >= args.max_steps:
            break
    model.load_state_dict(torch.load(f"{out}/model.pt"))
    results = {"args": vars(args), "scale": scale.tolist()}
    for sp in ["test", "test_heldout"]:
        d = splits[sp]
        if args.test_limit:
            d = {k: v[: args.test_limit] for k, v in d.items()}
        results[sp] = evaluate(model, args.model, data, d, scale, device, args.eval_rotations)
        say(sp, json.dumps(results[sp]))
        if args.model == "bc":
            results[sp + "_diag"] = diagnose(model, data, d, device)
            say(sp, "diag", json.dumps(results[sp + "_diag"]))
    json.dump(results, open(f"{out}/results.json", "w"), indent=2)


if __name__ == "__main__":
    main()

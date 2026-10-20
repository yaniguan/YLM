"""Train / evaluate the l=2 quadrupole question (src/quad.py) with the rotation protocol.

  python scripts/18_train_quad.py --model quad   --name quad_s0   # equivariant l=2 head
  python scripts/18_train_quad.py --model scalar --name quads_s0  # invariants only -> MLP (not equivariant)
Needs data/tensorqa_quad.json (17_make_quad_questions.py) and data/text_quad_matscibert.pt
(03_embed_text.py --name matscibert --questions data/tensorqa_quad.json --out data/text_quad_matscibert.pt).
Metrics are on the 5 coefficients of the 2e tensor; eq_err = |f(R x) - D(R) f(x)| / |f(x)|.
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
import torch.nn as nn
from e3nn import o3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.encoder import Encoder  # noqa: E402
from src.irreps_to_text import invariants  # noqa: E402
from src.quad import QuadModel, quad_to_2e  # noqa: E402

N_MAX = 29


class ScalarQuad(nn.Module):
    """Invariant features of the atoms, summed, then an MLP to 5 numbers. Cannot be equivariant."""

    def __init__(self, text_dim, hidden_irreps="64x0e + 8x0o + 32x1o + 8x1e + 16x2e", n_layers=3):
        super().__init__()
        self.encoder = Encoder(hidden_irreps=hidden_irreps, n_layers=n_layers)
        n = self.encoder.irreps_out.num_irreps
        self.mlp = nn.Sequential(nn.Linear(n, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 5))

    def forward(self, Z, pos, atom_mask, text, text_mask):
        x = invariants(self.encoder(Z, pos, atom_mask), self.encoder.irreps_out)
        return self.mlp((x * atom_mask[..., None]).sum(1))


def load():
    labels = pickle.load(open("data/qm9_xtb.pkl", "rb"))
    qs = json.load(open("data/tensorqa_quad.json"))
    cache = torch.load("data/text_quad_matscibert.pt")
    strings = sorted(cache["states"])
    T = max(s.shape[0] for s in cache["states"].values())
    text = torch.zeros(len(strings), T, cache["dim"])
    tmask = torch.zeros(len(strings), T, dtype=torch.bool)
    for k, s in enumerate(strings):
        st = cache["states"][s].float()
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
        Q = torch.tensor([q["target"] for q in sel], dtype=torch.float32)
        splits[sp] = dict(mol=torch.tensor([q["mol"] for q in sel]), text=torch.tensor([sid[q["question"]] for q in sel]),
                          Q=Q, y=quad_to_2e(Q))
    return dict(Z=Z, pos=pos, mask=mask, text=text, tmask=tmask, dim=cache["dim"]), splits


def batch(data, sp, idx, device, R=None):
    m = sp["mol"][idx]
    pos = data["pos"][m]
    if R is not None:
        pos = pos @ R.T
    t = sp["text"][idx]
    return (data["Z"][m].to(device), pos.to(device), data["mask"][m].to(device),
            data["text"][t].to(device), data["tmask"][t].to(device))


@torch.no_grad()
def evaluate(model, data, sp, scale, device, n_rot, bs=256):
    model.eval()
    rots = [torch.eye(3)]
    if n_rot > 1:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            rots += list(o3.rand_matrix(n_rot - 1))
    ae, cos, eq, rel_num, rel_den = [], [], [], 0.0, 0.0
    for s in range(0, len(sp["mol"]), bs):
        idx = torch.arange(s, min(s + bs, len(sp["mol"])))
        y0 = None
        for R in rots:
            D = o3.Irreps("2e").D_from_matrix(R)
            y = (model(*batch(data, sp, idx, device, R)) * scale).cpu()
            tgt = sp["y"][idx] @ D.T
            if y0 is None:
                y0 = y
            else:
                eq += ((y - y0 @ D.T).norm(dim=-1) / y0.norm(dim=-1).clamp(min=1e-12)).tolist()
            ae += (y - tgt).abs().mean(-1).tolist()
            cos += torch.nn.functional.cosine_similarity(y, tgt, dim=-1).tolist()
            rel_num += (y - tgt).pow(2).sum().item()
            rel_den += tgt.pow(2).sum().item()
    model.train()
    return dict(mae=float(np.mean(ae)), cosine=float(np.mean(cos)), rel_rmse=math.sqrt(rel_num / rel_den),
                eq_err_mean=float(np.mean(eq)) if eq else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["quad", "scalar"], required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--augment", action="store_true", help="random rotation per batch (needed for the scalar baseline to try)")
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

    data, splits = load()
    train = splits["train"]
    scale = train["y"].pow(2).mean().sqrt()
    model = (QuadModel(data["dim"]) if args.model == "quad" else ScalarQuad(data["dim"])).to(device)
    say("device", device, "| params", sum(q.numel() for q in model.parameters()), "| scale", scale.item(),
        "| n_train", len(train["mol"]))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total = args.epochs * math.ceil(len(train["mol"]) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    best, step, t0 = float("inf"), 0, time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(len(train["mol"]))
        ls = []
        for s in range(0, len(perm), args.batch_size):
            idx = perm[s: s + args.batch_size]
            R = o3.rand_matrix() if args.augment else None
            D = o3.Irreps("2e").D_from_matrix(R) if R is not None else None
            tgt = train["y"][idx] if D is None else train["y"][idx] @ D.T
            loss = ((model(*batch(data, train, idx, device, R)) - (tgt / scale).to(device)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            ls.append(loss.item())
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        val = evaluate(model, data, splits["val"], scale.to(device), device, 1)
        if val["mae"] < best:
            best = val["mae"]
            torch.save(model.state_dict(), f"{out}/model.pt")
        say(f"epoch {ep} loss {np.mean(ls):.4f} | val mae {val['mae']:.4f} rel_rmse {val['rel_rmse']:.3f} cos {val['cosine']:.3f}"
            f" | {time.time() - t0:.0f}s")
        if args.max_steps and step >= args.max_steps:
            break
    model.load_state_dict(torch.load(f"{out}/model.pt"))
    results = {"args": vars(args), "scale": scale.item()}
    for sp in ["test", "test_heldout"]:
        d = splits[sp]
        if args.test_limit:
            d = {k: v[: args.test_limit] for k, v in d.items()}
        results[sp] = evaluate(model, data, d, scale.to(device), device, args.eval_rotations)
        say(sp, json.dumps(results[sp]))
    json.dump(results, open(f"{out}/results.json", "w"), indent=2)


if __name__ == "__main__":
    main()

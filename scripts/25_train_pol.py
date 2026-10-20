"""TensorQA-pol: language-conditioned polarizability questions (LR-CCSD/d-aug-cc-pVDZ labels, QM7b).

  python scripts/25_train_pol.py --name pol_s0 --seed 0

One network predicts the molecule's polarizability tensor (0e + 2e); every question is an (task, atom A, atom B) program
over that tensor, executed exactly (src/pol.py). Training is teacher-forced with the true program, and the program heads
(task classifier + two atom pointers) are trained with cross-entropy. At evaluation the SAME tensor is queried with
  oracle   the true program
  learned  the program predicted by the heads
  rule     the program from the hand-written rule parser (data/parsed_pol.json, scripts/26_parse_pol.py)
and `true_tensor` executes the LABEL tensor with the learned/rule program (the program-only error).
`b3lyp` answers with the B3LYP tensor of the same molecule (a classical quantum-chemistry reference).
rel = RMSE / RMSE of the train-mean (iso, dir) or zero (aniso, ind) predictor.
"""
import argparse
import importlib.util
import json
import math
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import torch
from e3nn import o3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pol import CART, DIMS, OPS, PolModel, execute_pol  # noqa: E402

spec = importlib.util.spec_from_file_location("t22", os.path.join(os.path.dirname(os.path.abspath(__file__)), "22_train_v1.py"))
t22 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t22)
TASK = {t: i for i, t in enumerate(OPS)}


def tensors(labels, key):
    a = torch.tensor(np.stack([l[key] for l in labels]), dtype=torch.float32)
    return a.diagonal(dim1=-2, dim2=-1).sum(-1) / 3, CART.from_cartesian(a)[..., 1:]


def load(args, device):
    labels = pickle.load(open("data/qm7b_pol.pkl", "rb"))
    qs = json.load(open(args.data))
    parsed = json.load(open(args.parsed))
    enc = t22.TextEnc(args.text, device)
    strings = sorted({q["question"] for q in qs})
    uid = {s: k for k, s in enumerate(strings)}
    ids, att = enc.tokenize(strings)
    M, N = len(labels), 29
    Z, pos, mask = torch.zeros(M, N, dtype=torch.long), torch.zeros(M, N, 3), torch.zeros(M, N, dtype=torch.bool)
    for m, l in enumerate(labels):
        n = len(l["numbers"])
        Z[m, :n], pos[m, :n], mask[m, :n] = torch.tensor(l["numbers"]), torch.tensor(l["pos"], dtype=torch.float32), True
    a0, a2 = tensors(labels, "alpha")
    b0, b2 = tensors(labels, "alpha_b3lyp")
    data = dict(Z=Z, pos=pos, mask=mask, a0=a0, a2=a2, b0=b0, b2=b2, ids=ids, att=att, enc=enc)
    rows = {}
    for r, q in enumerate(qs):
        rows.setdefault(q["split"], []).append(r)
    splits = {}
    for sp, idx in rows.items():
        sel = [qs[r] for r in idx]
        tgt = torch.zeros(len(sel), 5)
        for i, q in enumerate(sel):
            tgt[i, : len(q["target"])] = torch.tensor(q["target"])
        splits[sp] = dict(row=torch.tensor(idx), mol=torch.tensor([q["mol"] for q in sel]),
                          uid=torch.tensor([uid[q["question"]] for q in sel]), task=torch.tensor([TASK[q["task"]] for q in sel]),
                          atoms=torch.tensor([q["atoms"] for q in sel]), target=tgt, style=[q["style"] for q in sel],
                          ptask=torch.tensor([TASK.get(parsed[r]["task"], -1) for r in idx]),
                          patoms=torch.tensor([parsed[r]["atoms"] for r in idx]))
    return data, splits


def get_batch(data, sp, idx, device, R=None, texts=None):
    m = sp["mol"][idx]
    pos = data["pos"][m]
    if R is not None:
        pos = pos @ R.T
    b = dict(Z=data["Z"][m].to(device), pos=pos.to(device), mask=data["mask"][m].to(device), task=sp["task"][idx].to(device),
             atoms=sp["atoms"][idx].to(device), tgt=sp["target"][idx].to(device), ptask=sp["ptask"][idx].to(device),
             patoms=sp["patoms"][idx].to(device), a0=data["a0"][m].to(device), a2=data["a2"][m].to(device),
             b0=data["b0"][m].to(device), b2=data["b2"][m].to(device))
    if texts is None:
        u = sp["uid"][idx]
        texts = data["enc"](data["ids"][u], data["att"][u])
    b["text"], b["tmask"] = texts
    return b


def dim_mask(task, device):
    d = torch.tensor([DIMS[t] for t in OPS], device=device)[task]
    return torch.arange(5, device=device)[None] < d[:, None]


def transform_expected(y, task, R):
    """how the answer must change when the molecule is rotated/reflected by R"""
    out = y.clone()
    D = o3.Irreps("2e").D_from_matrix(R).to(y.device)
    out[task == 1] = y[task == 1] @ D.T
    sel = task == 3
    out[sel, :3] = y[sel, :3] @ R.T.to(y.device)
    return out


@torch.no_grad()
def evaluate(model, data, sp, device, n_rot, base, bs=256):
    model.eval()
    n = len(sp["mol"])
    modes = ["oracle", "learned", "rule", "true_oracle", "true_learned", "true_rule", "b3lyp"]
    pred = {m: torch.zeros(n, 5) for m in modes}
    prog = {"learned": [[], [], []], "rule": [[], [], []]}
    eq = {"rot": [], "imp": []}
    with torch.random.fork_rng():
        torch.manual_seed(0)
        Rs = list(o3.rand_matrix(n_rot)) if n_rot else []
    for s in range(0, n, bs):
        idx = torch.arange(s, min(s + bs, n))
        b = get_batch(data, sp, idx, device)
        a0, a2, tl, ptr = model(b["Z"], b["pos"], b["mask"], b["text"], b["tmask"])
        lt, la = tl.argmax(-1), torch.stack([p.argmax(-1) for p in ptr], -1)
        progs = dict(oracle=(b["task"], b["atoms"]), learned=(lt, la), rule=(b["ptask"], b["patoms"]))
        for name, (pt, pa) in progs.items():
            pred[name][idx] = execute_pol(a0, a2, b["pos"], pt, pa).cpu()
            pred["true_" + name][idx] = execute_pol(b["a0"], b["a2"], b["pos"], pt, pa).cpu()
            if name != "oracle":
                need = b["task"] >= 2
                prog[name][0] += (pt == b["task"]).cpu().tolist()
                prog[name][1] += ((pa[:, 0] == b["atoms"][:, 0]) | ~need).cpu().tolist()
                prog[name][2] += ((pa[:, 1] == b["atoms"][:, 1]) | ~need).cpu().tolist()
        pred["b3lyp"][idx] = execute_pol(b["b0"], b["b2"], b["pos"], b["task"], b["atoms"]).cpu()
        y0 = execute_pol(a0, a2, b["pos"], b["task"], b["atoms"])
        for kind, sign in (("rot", 1), ("imp", -1)):
            for R in Rs:
                R = sign * R
                bb = get_batch(data, sp, idx, device, R, (b["text"], b["tmask"]))
                c0, c2, _, _ = model(bb["Z"], bb["pos"], bb["mask"], bb["text"], bb["tmask"])
                y = execute_pol(c0, c2, bb["pos"], bb["task"], bb["atoms"])
                e = (y - transform_expected(y0, b["task"], R)).norm(dim=-1) / y0.norm(dim=-1).clamp(min=1e-12)
                eq[kind] += e.cpu().tolist()
    model.train()
    task, tgt = sp["task"], sp["target"]
    out = dict(n=n)
    for t, k in TASK.items():
        sel = task == k
        d = DIMS[t]
        T = tgt[sel][:, :d]
        base_pred = torch.full_like(T, base[t]) if t in ("pol_iso", "pol_dir") else torch.zeros_like(T)
        rmse_base = (T - base_pred).pow(2).sum(-1).mean().sqrt().item()
        res = {}
        for mname in modes:
            P = pred[mname][sel][:, :d]
            r = dict(mae=(P - T).abs().mean().item(), rel=(P - T).pow(2).sum(-1).mean().sqrt().item() / rmse_base)
            if d > 1:
                r["cosine"] = torch.nn.functional.cosine_similarity(P, T, dim=-1).mean().item()
            res[mname] = r
        out[t] = res
    for name in ("learned", "rule"):
        a = [np.mean(v) for v in prog[name]]
        out["program_" + name] = dict(task_acc=a[0], atom_a_acc=a[1], atom_b_acc=a[2])
    for k, name in (("rot", "rotation"), ("imp", "reflection")):
        if eq[k]:
            out["eq_err_" + name] = float(np.mean(eq[k]))
    styles = np.array(sp["style"])
    err = (pred["learned"] - tgt).pow(2).sum(-1).sqrt().numpy()
    out["by_style_learned_abs_err"] = {s: float(err[styles == s].mean()) for s in sorted(set(styles))}
    rec = {m: (pred[m] - tgt).pow(2).sum(-1).sqrt().numpy() for m in modes}
    rec.update(row=sp["row"].numpy(), mol=sp["mol"].numpy(), task=task.numpy())
    return out, rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--data", default="data/tensorqa_pol.json")
    p.add_argument("--parsed", default="data/parsed_pol.json")
    p.add_argument("--text", choices=list(t22.TEXT), default="matscibert")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--prog_w", type=float, default=1.0)
    p.add_argument("--eval_rotations", type=int, default=2)
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

    data, splits = load(args, device)
    train = splits["train"]
    ttask = train["task"]
    iso_mu, iso_sd = train["target"][ttask == 0, 0].mean().item(), train["target"][ttask == 0, 0].std().item()
    base = {"pol_iso": iso_mu, "pol_dir": train["target"][ttask == 2, 0].mean().item()}
    scale = torch.tensor([train["target"][ttask == k][:, :DIMS[t]].std().item() if t in ("pol_iso", "pol_dir") else
                          train["target"][ttask == k][:, :DIMS[t]].pow(2).mean().sqrt().item() for t, k in TASK.items()], device=device)
    s2 = train["target"][ttask == 1].pow(2).mean().sqrt().item()
    mean_atoms = data["mask"][train["mol"]].sum(1).float().mean().item()
    model = PolModel(data["enc"].dim, iso_mean=iso_mu, iso_std=iso_sd, s2=s2, mean_atoms=mean_atoms).to(device)
    say("device", device, "| params", sum(q.numel() for q in model.parameters()), "| scale", scale.tolist(), "| n_train", len(train["mol"]))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total = args.epochs * math.ceil(len(train["mol"]) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    ce = torch.nn.functional.cross_entropy
    best, step, t0 = float("inf"), 0, time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(len(train["mol"]))
        ls = []
        for s in range(0, len(perm), args.batch_size):
            b = get_batch(data, train, perm[s: s + args.batch_size], device)
            a0, a2, tl, ptr = model(b["Z"], b["pos"], b["mask"], b["text"], b["tmask"])
            y = execute_pol(a0, a2, b["pos"], b["task"], b["atoms"])
            dm = dim_mask(b["task"], device)
            loss = ((((y - b["tgt"]) / scale[b["task"]][:, None]) ** 2) * dm).sum() / dm.sum()
            need = b["task"] >= 2
            pl = ce(tl, b["task"])
            if need.any():
                pl = pl + ce(ptr[0][need], b["atoms"][need, 0]) + ce(ptr[1][need], b["atoms"][need, 1])
            loss = loss + args.prog_w * pl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            ls.append(loss.item())
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        vs = splits["val"] if not args.test_limit else {k: (v[: args.test_limit] if torch.is_tensor(v) else v[: args.test_limit]) for k, v in splits["val"].items()}
        val, _ = evaluate(model, data, vs, device, 0, base)
        score = sum(val[t]["oracle"]["rel"] for t in OPS)
        if score < best:
            best = score
            torch.save(model.state_dict(), f"{out}/model.pt")
        say(f"epoch {ep} loss {np.mean(ls):.4f} | val rel(oracle) " + " ".join(f"{t[4:]} {val[t]['oracle']['rel']:.3f}" for t in OPS)
            + f" | prog acc {val['program_learned']['task_acc']:.3f} | {time.time() - t0:.0f}s")
        if args.max_steps and step >= args.max_steps:
            break
    model.load_state_dict(torch.load(f"{out}/model.pt"))
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    results = dict(args=vars(args), git=git, data_sha=t22.sha(args.data), parsed_sha=t22.sha(args.parsed), base=base, scale=scale.tolist())
    records = {}
    for sp_name in ["test", "test_heldout"]:
        d = splits[sp_name]
        if args.test_limit:
            d = {k: (v[: args.test_limit] if torch.is_tensor(v) else v[: args.test_limit]) for k, v in d.items()}
        m, rec = evaluate(model, data, d, device, args.eval_rotations, base)
        results[sp_name] = m
        records.update({f"{sp_name}/{k}": v for k, v in rec.items()})
        say(sp_name, json.dumps({k: v for k, v in m.items() if k != "by_style_learned_abs_err"}))
    json.dump(results, open(f"{out}/results.json", "w"), indent=1)
    np.savez_compressed(f"{out}/records.npz", **records)


if __name__ == "__main__":
    main()

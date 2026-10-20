"""Train / evaluate on TensorQA-v1 (force, diff, proj; free-form wordings; systematic-generalisation splits).

  python scripts/22_train_v1.py --model bc     --name v1_bc_s0       # Option B/C-lite, text -> coefficients
  python scripts/22_train_v1.py --model optA   --name v1_optA_s0     # Option A: one pointer, one vector
  python scripts/22_train_v1.py --model prog   --name v1_prog_s0     # learned parser (program supervision) + exact executor
  python scripts/22_train_v1.py --model rule   --name v1_rule_s0     # hand-written rule parser + exact executor (needs data/parsed_v1.json)
  python scripts/22_train_v1.py --model oracle --name v1_oracle_s0   # ground-truth program + exact executor (diagnostic)

The question is encoded ONLINE by a frozen text encoder (free-form wordings make a per-question cache too big).
Everything is reported per split and per task; for the 2x2 diagnosis see `program` and `true_forces` in results.json.
Controls: --parity {so3,o3}, --aux_force, --n_mols_frac / --q_per_mol_frac (physical vs language budget),
--train_styles / --fix_total (wording coverage at a fixed number of training questions).
"""
import argparse
import hashlib
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
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.baselines import PipelineComp, execute  # noqa: E402
from src.model import YLM  # noqa: E402
from src.option_bc import CompModel  # noqa: E402
from src.prog import ProgModel  # noqa: E402

TASKS = {"force": 0, "diff": 1, "proj": 2}
N_MAX = 29
TEXT = {"matscibert": "m3rg-iitd/matscibert", "minilm": "sentence-transformers/all-MiniLM-L6-v2"}
EVAL_SPLITS = ["val", "test", "test_heldout", "scaf", "scaf_heldout", "comb", "cf"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


class TextEnc:
    """Frozen text encoder run on the fly."""

    def __init__(self, name, device):
        self.tok = AutoTokenizer.from_pretrained(TEXT[name])
        self.model = AutoModel.from_pretrained(TEXT[name]).eval().to(device)
        for q in self.model.parameters():
            q.requires_grad_(False)
        self.dim, self.device = self.model.config.hidden_size, device

    @torch.no_grad()
    def tokenize(self, strings, max_len=128):
        enc = self.tok(strings, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"].bool()

    @torch.no_grad()
    def __call__(self, ids, att):
        L = int(att.sum(1).max())
        ids, att = ids[:, :L].to(self.device), att[:, :L].to(self.device)
        with torch.autocast(self.device.split(":")[0], dtype=torch.float16, enabled=self.device != "cpu"):
            h = self.model(input_ids=ids, attention_mask=att).last_hidden_state
        return h.float(), att


def load(args, device):
    labels = pickle.load(open("data/qm9_xtb.pkl", "rb"))
    qs = json.load(open(args.data))
    parsed = json.load(open(args.parsed)) if args.model == "rule" else None
    enc = TextEnc(args.text, device)
    strings = sorted({q["question"] for q in qs})
    uid_of = {s: k for k, s in enumerate(strings)}
    ids, att = enc.tokenize(strings)
    M = len(labels)
    Z = torch.zeros(M, N_MAX, dtype=torch.long)
    pos = torch.zeros(M, N_MAX, 3)
    mask = torch.zeros(M, N_MAX, dtype=torch.bool)
    forces = torch.zeros(M, N_MAX, 3)
    for m, lab in enumerate(labels):
        n = len(lab["numbers"])
        Z[m, :n] = torch.tensor(lab["numbers"])
        pos[m, :n] = torch.tensor(lab["pos"], dtype=torch.float32)
        forces[m, :n] = torch.tensor(lab["forces"], dtype=torch.float32)
        mask[m, :n] = True
    rows = {}
    for r, q in enumerate(qs):
        rows.setdefault(q["split"], []).append(r)
    splits = {}
    for sp, idx in rows.items():
        sel = [qs[r] for r in idx]
        d = dict(row=torch.tensor(idx), mol=torch.tensor([q["mol"] for q in sel]),
                 uid=torch.tensor([uid_of[q["question"]] for q in sel]),
                 task=torch.tensor([TASKS[q["task"]] for q in sel]),
                 atoms=torch.tensor([q["atoms"] for q in sel]),
                 target=torch.tensor([q["target"] for q in sel], dtype=torch.float32),
                 style=[q.get("style", "") for q in sel])
        if "gid" in sel[0]:
            d["gid"] = torch.tensor([q["gid"] for q in sel])
            d["kind"] = [q["kind"] for q in sel]
            d["k"] = torch.tensor([q["k"] for q in sel])
        if parsed is not None:
            d["ptask"] = torch.tensor([TASKS.get(parsed[r]["task"], -1) for r in idx])
            d["patoms"] = torch.tensor([parsed[r]["atoms"] for r in idx])
        splits[sp] = d
    return dict(Z=Z, pos=pos, mask=mask, forces=forces, ids=ids, att=att, enc=enc), splits, qs


def subset(sp, keep):
    keep = torch.as_tensor(keep)
    out = {}
    for k, v in sp.items():
        out[k] = v[keep] if torch.is_tensor(v) else [v[i] for i in keep.tolist()]
    return out


def restrict_train(args, train, qs):
    """Budget controls. Physical budget = labelled molecules; language budget = questions per molecule / styles."""
    rng = np.random.RandomState(1234)
    keep = np.arange(len(train["mol"]))
    if args.n_mols_frac < 1.0:
        mols = np.unique(train["mol"].numpy())
        chosen = set(rng.choice(mols, max(1, int(args.n_mols_frac * len(mols))), replace=False).tolist())
        keep = np.array([i for i in keep if train["mol"][i].item() in chosen])
    if args.q_per_mol_frac < 1.0:
        keep = keep[rng.rand(len(keep)) < args.q_per_mol_frac]
    if args.train_styles:
        ok = set(args.train_styles.split(","))
        keep = np.array([i for i in keep if train["style"][i] in ok])
    if args.fix_total:
        keep = rng.choice(keep, args.fix_total, replace=len(keep) < args.fix_total)
    return subset(train, keep)


def get_batch(data, sp, idx, device, R=None, texts=None):
    m = sp["mol"][idx]
    pos, tgt, frc = data["pos"][m], sp["target"][idx], data["forces"][m]
    if R is not None:
        pos, tgt, frc = pos @ R.T, tgt @ R.T, frc @ R.T
    b = dict(Z=data["Z"][m].to(device), pos=pos.to(device), mask=data["mask"][m].to(device),
             task=sp["task"][idx].to(device), atoms=sp["atoms"][idx].to(device),
             forces=frc.to(device), tgt=tgt.to(device))
    if "ptask" in sp:
        b["ptask"], b["patoms"] = sp["ptask"][idx].to(device), sp["patoms"][idx].to(device)
    if texts is None:
        texts = data["enc"](data["ids"][sp["uid"][idx]], data["att"][sp["uid"][idx]])
    b["text"], b["tmask"] = texts
    return b


def forward(model, kind, b, scale, ops):
    """-> y in units of the per-task RMS target, aux dict"""
    aux = {}
    if kind == "bc":
        y = model(b["Z"], b["pos"], b["mask"], b["text"], b["tmask"])[0]
    elif kind == "optA":
        y = model(b["Z"], b["pos"], b["mask"], b["text"], b["tmask"], b["task"])[0]
    elif kind == "prog":
        y, aux = model(b["Z"], b["pos"], b["mask"], b["text"], b["tmask"])
        y = y / scale[b["task"]][:, None]
    else:
        prog = (b["task"], b["atoms"]) if kind == "oracle" else (b["ptask"], b["patoms"])
        y = model(b["Z"], b["pos"], b["mask"], *prog) / scale[b["task"]][:, None]
    return y, aux


def predicted_program(kind, aux, b):
    if kind == "prog":
        t = aux["task_logits"].argmax(-1)
        a1, a2 = (lg.argmax(-1) for lg in aux["ptr_logits"])
        return t, torch.stack([a1, a2], -1)
    if kind == "oracle":
        return b["task"], b["atoms"]
    if kind == "rule":
        return b["ptask"], b["patoms"]
    return None


def cosine(y, t):
    return torch.nn.functional.cosine_similarity(y, t, dim=-1)


@torch.no_grad()
def evaluate(model, kind, data, sp, scale, device, n_rot, ops, bs=256, keep_pred=False):
    model.eval()
    rots = [("id", torch.eye(3))]
    if n_rot:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            Rs = list(o3.rand_matrix(n_rot))
        rots += [("rot", R) for R in Rs] + [("imp", -R) for R in Rs]
    n = len(sp["mol"])
    ae, cos, tf_ae, tf_cos = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    eq = {"rot": [], "imp": []}
    pred = np.zeros((n, 3))
    prog_ok = dict(task=[], a=[], b=[])
    for s in range(0, n, bs):
        idx = torch.arange(s, min(s + bs, n))
        texts = None
        y0 = None
        for name, R in rots:
            b = get_batch(data, sp, idx, device, R, texts)
            texts = (b["text"], b["tmask"])
            y, aux = forward(model, kind, b, scale, ops)
            y = y * scale[b["task"]][:, None]
            if name == "id":
                y0 = y
                tgt = b["tgt"]
                ae[idx] = (y - tgt).abs().mean(-1).cpu().numpy()
                cos[idx] = cosine(y, tgt).cpu().numpy()
                pred[idx] = y.cpu().numpy()
                p = predicted_program(kind, aux, b)
                if p is not None:   # 2x2 cell: TRUE forces, program as predicted
                    t_pred, a_pred = p
                    ytf = execute(ops, b["forces"], b["pos"], t_pred, a_pred)
                    tf_ae[idx] = (ytf - tgt).abs().mean(-1).cpu().numpy()
                    tf_cos[idx] = cosine(ytf, tgt).cpu().numpy()
                    prog_ok["task"] += (t_pred == b["task"]).cpu().tolist()
                    need_b = b["task"] != TASKS["force"]
                    prog_ok["a"] += (a_pred[:, 0] == b["atoms"][:, 0]).cpu().tolist()
                    prog_ok["b"] += ((a_pred[:, 1] == b["atoms"][:, 1]) | ~need_b).cpu().tolist()
            else:
                e = (y - y0 @ R.to(device).T).norm(dim=-1) / y0.norm(dim=-1).clamp(min=1e-12)
                eq[name] += e.cpu().tolist()
    model.train()
    out = dict(n=n)
    task = sp["task"].numpy()
    for t, k in TASKS.items():
        sel = task == k
        if sel.any():
            out[t] = dict(mae=float(ae[sel].mean()), cosine=float(cos[sel].mean()), n=int(sel.sum()))
            if prog_ok["task"]:
                out[t]["true_forces_mae"] = float(tf_ae[sel].mean())
                out[t]["true_forces_cosine"] = float(tf_cos[sel].mean())
    if prog_ok["task"]:
        ok = {k: np.array(v) for k, v in prog_ok.items()}
        out["program"] = dict(task_acc=float(ok["task"].mean()), atom_a_acc=float(ok["a"].mean()),
                              atom_b_acc=float(ok["b"].mean()),
                              exact=float((ok["task"] & ok["a"] & ok["b"]).mean()))
    for name in ("rot", "imp"):
        if eq[name]:
            out["eq_err_" + ("rotation" if name == "rot" else "reflection")] = float(np.mean(eq[name]))
    styles = np.array(sp["style"])
    if (styles != "").any():
        out["by_style"] = {st: dict(cosine=float(cos[styles == st].mean()), mae=float(ae[styles == st].mean()),
                                    n=int((styles == st).sum())) for st in sorted(set(styles)) if st}
    rec = dict(row=sp["row"].numpy(), ae=ae, cos=cos, mol=sp["mol"].numpy())
    if prog_ok["task"]:
        rec.update(tf_ae=tf_ae, tf_cos=tf_cos)
    return out, rec, (pred if keep_pred else None)


def cf_consistency(sp, pred):
    """Counterfactual identities on predicted D: antisymmetry D(A,B) = -D(B,A); chain D(A,C) = D(A,B) + D(B,C)."""
    res = {}
    tgt = sp["target"].numpy()
    for kind, order in [("swap", (0, 1)), ("chain", (0, 1, 2))]:
        for src, arr in [("model", pred), ("labels", tgt)]:
            errs = []
            rows = {}
            for i, (g, k, kd) in enumerate(zip(sp["gid"].tolist(), sp["k"].tolist(), sp["kind"])):
                if kd == kind:
                    rows.setdefault(g, {})[k] = i
            for g, d in rows.items():
                if len(d) != len(order):
                    continue
                y = [arr[d[k]] for k in order]
                if kind == "swap":
                    num, den = np.linalg.norm(y[0] + y[1]), np.linalg.norm(y[0]) + np.linalg.norm(y[1])
                else:
                    num, den = np.linalg.norm(y[2] - y[0] - y[1]), sum(np.linalg.norm(v) for v in y)
                errs.append(num / max(den, 1e-12))
            res[f"{kind}_{src}"] = float(np.mean(errs)) if errs else None
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["bc", "optA", "prog", "rule", "oracle"], required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--data", default="data/tensorqa_v1.json")
    p.add_argument("--parsed", default="data/parsed_v1.json")
    p.add_argument("--text", choices=list(TEXT), default="matscibert")
    p.add_argument("--parity", choices=["so3", "o3"], default="o3")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--prog_w", type=float, default=1.0, help="weight of program supervision (prog only)")
    p.add_argument("--aux_force", type=float, default=0.0, help="weight of direct per-atom force supervision (prog/rule/oracle)")
    p.add_argument("--token_dropout", type=float, default=0.0)
    p.add_argument("--n_mols_frac", type=float, default=1.0)
    p.add_argument("--q_per_mol_frac", type=float, default=1.0)
    p.add_argument("--train_styles", default="")
    p.add_argument("--fix_total", type=int, default=0)
    p.add_argument("--eval_rotations", type=int, default=2, help="proper rotations; the same number of reflections")
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--test_limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = f"runs/{args.name}"
    os.makedirs(out, exist_ok=True)
    log = open(f"{out}/log.txt", "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    data, splits, qs = load(args, device)
    ops = tuple(TASKS)
    train = restrict_train(args, splits["train"], qs)
    scale = torch.stack([train["target"][train["task"] == k].pow(2).mean().sqrt() for k in TASKS.values()]).to(device)
    dim = data["enc"].dim
    if args.model == "bc":
        model = CompModel(dim, token_dropout=args.token_dropout, parity=args.parity)
    elif args.model == "optA":
        model = YLM(text_dim=dim, parity=args.parity)
    elif args.model == "prog":
        model = ProgModel(dim, parity=args.parity)
    else:
        model = PipelineComp(ops=ops)
    model = model.to(device)
    say("device", device, "| params", sum(q.numel() for q in model.parameters()), "| scale", scale.tolist(),
        "| n_train", len(train["mol"]), "| n_train_mols", len(set(train["mol"].tolist())))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total = args.epochs * math.ceil(len(train["mol"]) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    best, step, t0 = float("inf"), 0, time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(len(train["mol"]))
        ls = []
        for s in range(0, len(perm), args.batch_size):
            b = get_batch(data, train, perm[s: s + args.batch_size], device)
            y, aux = forward(model, args.model, b, scale, ops)
            loss = ((y - b["tgt"] / scale[b["task"]][:, None]) ** 2).mean()
            if args.model == "prog" and args.prog_w > 0:
                ce = torch.nn.functional.cross_entropy
                has_b = b["atoms"][:, 1] >= 0
                prog_loss = ce(aux["task_logits"], b["task"]) + ce(aux["ptr_logits"][0], b["atoms"][:, 0])
                if has_b.any():
                    prog_loss = prog_loss + ce(aux["ptr_logits"][1][has_b], b["atoms"][has_b, 1])
                loss = loss + args.prog_w * prog_loss
            if args.aux_force > 0 and args.model in ("prog", "rule", "oracle"):
                V = model.last_V
                fl = (((V - b["forces"]) / scale[0]) ** 2).sum(-1)
                loss = loss + args.aux_force * (fl * b["mask"]).sum() / b["mask"].sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            ls.append(loss.item())
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        val_sp = splits["val"] if not args.test_limit else subset(splits["val"], torch.arange(min(args.test_limit, len(splits["val"]["mol"]))))
        val, _, _ = evaluate(model, args.model, data, val_sp, scale, device, 0, ops)
        score = sum(val[t]["mae"] / scale[k].item() for t, k in TASKS.items())
        if score < best:
            best = score
            torch.save(model.state_dict(), f"{out}/model.pt")
        say(f"epoch {ep} loss {np.mean(ls):.4f} | " + " | ".join(f"{t} mae {val[t]['mae']:.4f} cos {val[t]['cosine']:.3f}" for t in TASKS)
            + f" | {time.time() - t0:.0f}s")
        if args.max_steps and step >= args.max_steps:
            break
    model.load_state_dict(torch.load(f"{out}/model.pt"))
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    results = dict(args=vars(args), scale=scale.tolist(), git=git, data_sha=sha(args.data),
                   parsed_sha=sha(args.parsed) if args.model == "rule" else None, n_train=len(train["mol"]))
    records = {}
    for sp_name in EVAL_SPLITS[1:]:
        if sp_name not in splits:
            continue
        d = splits[sp_name]
        if args.test_limit:
            d = subset(d, torch.arange(min(args.test_limit, len(d["mol"]))))
        m, rec, pred = evaluate(model, args.model, data, d, scale, device, args.eval_rotations, ops,
                                keep_pred=sp_name == "cf")
        if sp_name == "cf" and len(d["mol"]) == len(splits["cf"]["mol"]):
            m["consistency"] = cf_consistency(d, pred)
        results[sp_name] = m
        records.update({f"{sp_name}/{k}": v for k, v in rec.items()})
        say(sp_name, json.dumps({k: v for k, v in m.items() if k != "by_style"}))
    json.dump(results, open(f"{out}/results.json", "w"), indent=1)
    np.savez_compressed(f"{out}/records.npz", **records)


if __name__ == "__main__":
    main()

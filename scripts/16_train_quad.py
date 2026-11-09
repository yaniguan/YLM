"""Train the opt-in 0e+2e readout on molecular quadrupole labels.

Does not change TensorQA-v0 training or the default vector head.
Needs data/qm9_quad.pkl from scripts/12_make_quadrupole.py and data/qm9_xtb.pkl.

  python scripts/16_train_quad.py --name quad_s0
  python scripts/16_train_quad.py --name quad_tiny --tiny --epochs 1 --max_steps 1 --device cpu

The question is global (the molecular quadrupole), so the text side is a learned
task token. Option A is the default. Pass --option B or C to use those heads.
Splits follow data/tensorqa_v0.json when that file is present. Otherwise the
split is molecule index mod 10: 8 is val, 9 is test, and the rest is train.
Quadrupole labels can include molecules that the question file skipped (a force
above 0.5 hartree/bohr). Those molecules are dropped. They are not used for test.
"""
import argparse
import json
import math
import os
import pickle
import sys
import time

import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src  # noqa: E402
from e3nn import o3

from src.model import YLM  # noqa: E402
from src.tensor_readout import (  # noqa: E402
    build_arrays,
    load_quadrupole_labels,
    make_batch,
    quadrupole_loss,
    split_rows,
    tensor_equivariance_error,
    unique_entries,
)


def splits_from_questions(mol_ids, questions_path):
    """Return row splits and how many molecules were dropped.

    When the question file exists, a molecule with no train, val, or test
    split is dropped. Dropped molecules are not used for test.
    When the question file is missing, every molecule is kept. The split is
    then molecule index mod 10: 8 is val, 9 is test, and the rest is train.
    """
    if not questions_path or not os.path.exists(questions_path):
        return split_rows(mol_ids), 0
    with open(questions_path) as handle:
        questions = json.load(handle)
    split_of = {}
    for question in questions:
        if question["split"] != "test_heldout":
            split_of[question["mol"]] = question["split"]
    rows = {"train": [], "val": [], "test": []}
    dropped = 0
    for row, mol in enumerate(mol_ids.tolist()):
        name = split_of.get(int(mol))
        if name not in rows:
            dropped += 1
            continue
        rows[name].append(row)
    splits = {name: torch.tensor(index, dtype=torch.long) for name, index in rows.items()}
    return splits, dropped


@torch.no_grad()
def evaluate(model, arrays, rows, scale, device, n_rot, batch_size=32):
    """MAE on the six unique entries, plus the rotation error of the 3x3 matrix."""
    model.eval()
    rotations = [torch.eye(3)]
    if n_rot > 1:
        with torch.random.fork_rng():
            torch.manual_seed(0)
            rotations += list(o3.rand_matrix(n_rot - 1))
    abs_err, eq_err = [], []
    scale = scale.to(device)
    eye = torch.eye(3, dtype=scale.dtype)
    for start in range(0, len(rows), batch_size):
        idx = rows[start:start + batch_size]
        y0 = None
        for R in rotations:
            use_R = None if torch.equal(R, eye) else R
            inputs, target = make_batch(arrays, idx, device, use_R)
            pred = (model(*inputs)[0] * scale).cpu()
            target = target.cpu()
            if y0 is None:
                y0 = pred
            else:
                eq_err += tensor_equivariance_error(pred, y0, R).tolist()
            abs_err += (unique_entries(pred) - unique_entries(target)).abs().mean(-1).tolist()
    model.train()
    return {
        "n": int(len(rows)),
        "mae": float(sum(abs_err) / len(abs_err)),
        "eq_err_mean": float(sum(eq_err) / len(eq_err)) if eq_err else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--geom", default="data/qm9_xtb.pkl")
    p.add_argument("--quad", default="data/qm9_quad.pkl")
    p.add_argument("--questions", default="data/tensorqa_v0.json")
    p.add_argument("--option", choices=list("ABC"), default="A")
    p.add_argument("--tiny", action="store_true", help="small model for a CPU smoke run")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval_rotations", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = f"runs/{args.name}"
    os.makedirs(out_dir, exist_ok=True)
    log = open(f"{out_dir}/log.txt", "w")

    def say(*parts):
        line = " ".join(str(part) for part in parts)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    if not os.path.exists(args.quad):
        raise SystemExit(f"Missing {args.quad}. Build it with scripts/12_make_quadrupole.py.")
    if not os.path.exists(args.geom):
        raise SystemExit(f"Missing {args.geom}.")
    with open(args.geom, "rb") as handle:
        geom = pickle.load(handle)
    quad = load_quadrupole_labels(args.quad)
    arrays = build_arrays(geom, quad)
    splits, n_dropped = splits_from_questions(arrays["mol"], args.questions)
    say("molecules dropped (no question split):", n_dropped)
    for name, rows in splits.items():
        if len(rows) == 0:
            raise SystemExit(f"Split {name} is empty. Check the quadrupole file and the question splits.")
    train_rows = splits["train"]
    scale = unique_entries(arrays["target"][train_rows]).pow(2).mean().sqrt().clamp(min=1e-8)
    say("device", device, "| molecules", len(arrays["mol"]), "| scale", float(scale))

    tiny = {}
    if args.tiny:
        tiny = dict(hidden_irreps="8x0e + 2x0o + 4x1o + 2x1e + 2x2e",
                    cross_irreps="4x0e + 4x1o + 2x1e + 2x2e")
    model = YLM(text_dim=16, text_mode="taskid", n_tasks=1, option=args.option,
                tensor_readout=True, **tiny).to(device)
    say("parameters", sum(param.numel() for param in model.parameters()), "| option", args.option)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    scale_dev = scale.to(device)
    best, step, t0 = float("inf"), 0, time.time()
    for epoch in range(args.epochs):
        perm = train_rows[torch.randperm(len(train_rows))]
        losses = []
        for start in range(0, len(perm), args.batch_size):
            inputs, target = make_batch(arrays, perm[start:start + args.batch_size], device)
            pred = model(*inputs)[0]
            loss = quadrupole_loss(pred, target / scale_dev)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        val = evaluate(model, arrays, splits["val"], scale, device, n_rot=1)
        if val["mae"] < best:
            best = val["mae"]
            torch.save(model.state_dict(), f"{out_dir}/model.pt")
        say(f"epoch {epoch} loss {sum(losses) / len(losses):.4f} | val mae {val['mae']:.6f} | {time.time() - t0:.0f}s")
        if args.max_steps and step >= args.max_steps:
            break
    model.load_state_dict(torch.load(f"{out_dir}/model.pt", map_location=device))
    results = {"args": vars(args), "scale": float(scale), "n_dropped": n_dropped}
    for name in ("val", "test"):
        results[name] = evaluate(model, arrays, splits[name], scale, device, n_rot=args.eval_rotations)
        say(name, json.dumps(results[name]))
    with open(f"{out_dir}/results.json", "w") as handle:
        json.dump(results, handle, indent=2)
    say("saved", out_dir)


if __name__ == "__main__":
    main()

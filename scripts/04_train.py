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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src
from e3nn import o3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.profiling import Profiler
from src.model import YLM  # noqa: E402
from src.attn_supervision import attention_supervision_loss  # noqa: E402

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
    inputs = (data["Z"][m].to(device), pos.to(device), data["mask"][m].to(device),
              data["text"][t].to(device), data["text_mask"][t].to(device),
              split["task"][idx].to(device))
    if data.get("with_atom"):  # pipeline baseline: the atom index the parser found (or the true one in training)
        inputs += (split.get("atom_in", split["atom"])[idx].to(device),)
    return inputs, target.to(device)


@torch.no_grad()
def evaluate(model, data, split, scale, device, n_rot, batch_size=512, seed=0, reflection=False, reflection_only=False, routing_audit=False):
    model.eval()
    from src.routing_metrics import RoutingAudit
    audit = RoutingAudit() if routing_audit else None
    cross = getattr(model, "cross", None)
    if audit is not None and cross is None:
        raise ValueError("Routing audit requires a YLM cross-attention model")
    if cross is not None:
        cross.audit_enabled = routing_audit
    scale = scale.to(device)
    rotations = [torch.eye(3)]  # first one = no rotation
    if n_rot > 1:
        with torch.random.fork_rng():  # same rotations every time, without touching the training RNG
            torch.manual_seed(seed)
            rotations += list(o3.rand_matrix(n_rot - 1))
    if reflection:
        rotations += [-R for R in rotations.copy()]
    n = len(split["mol"])
    res = {t: {"abs_err": [], "cos": [], "eq_err": [], "atom_hit": []} for t in TASKS}
    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n))
        task = split["task"][idx]
        y0 = None
        for R in rotations:
            inputs, target = get_batch(data, split, idx, device, R)
            y, attn = model(*inputs)
            if audit is not None and torch.equal(R, torch.eye(3)):
                audit.update(idx.tolist(), attn, split["atom"][idx], task, inputs[2],
                             inputs[1], cross.last_scores, inputs[4])
            y = (y * scale[inputs[5]][:, None]).cpu()
            target = target.cpu()
            if y0 is None:
                y0 = y  # prediction in the original frame (R = identity)
            if reflection_only and torch.linalg.det(R) > 0:
                continue
            if not torch.equal(R, torch.eye(3)):
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
    if audit is not None:
        out["routing"] = audit.report()
    if cross is not None:
        cross.audit_enabled = False
        cross.last_scores = None
    model.train()
    return out


def attach_parsed_atoms(splits, sdf, labels_path, questions_path, say, cache_path="data/parsed_atoms.json"):
    """Pipeline baseline: replace the atom index of val/test splits by what the rule parser finds.

    The parse needs RDKit. It is cached in cache_path (one entry per question, -1 = not a force question
    or parse failure), so a machine without RDKit (e.g. an old cluster) can reuse a file made elsewhere."""
    questions = json.load(open(questions_path))
    if os.path.exists(cache_path):
        parsed_all = json.load(open(cache_path))
        assert len(parsed_all) == len(questions), "parsed_atoms.json does not match the questions file"
    else:
        from rdkit import Chem, RDLogger
        from src.parse_question import resolve
        RDLogger.DisableLog("rdApp.*")
        labels = pickle.load(open(labels_path, "rb"))
        supplier = Chem.SDMolSupplier(sdf, removeHs=False, sanitize=True)
        parsed_all = [resolve(q["question"], supplier[labels[q["mol"]]["qm9_index"]]) if q["task"] == "force" else -1
                      for q in questions]
        json.dump(parsed_all, open(cache_path, "w"))
    for name in ["val", "test", "test_heldout"]:
        idx = [k for k, q in enumerate(questions) if q["split"] == name]
        qs = [questions[k] for k in idx]
        parsed = torch.tensor([parsed_all[k] for k in idx], dtype=torch.long)
        force = [k for k, q in enumerate(qs) if q["task"] == "force"]
        ok = sum(int(parsed[k] == qs[k]["atom"]) for k in force)
        splits[name]["atom_in"] = parsed
        splits[name]["parse_acc"] = ok / max(len(force), 1)
        say(f"parser accuracy {name}: {ok}/{len(force)} = {splits[name]['parse_acc']:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", choices=["minilm", "matscibert", "taskid"], required=True)
    p.add_argument("--model", choices=["ylm", "pipeline", "scalar"], default="ylm",
                   help="ylm = Option A; pipeline = parse text -> atom index -> force model; "
                        "scalar = invariants only (see src/baselines.py)")
    p.add_argument("--routing_audit", action="store_true")
    p.add_argument("--option", choices=list("ABC"), default="A")
    p.add_argument("--legacy_odd_scalars", action="store_true",
                   help="ablation: raw odd scalars in keys; breaks reflection invariance")
    p.add_argument("--tiny", action="store_true", help="small three-layer CPU model")
    p.add_argument("--eval_reflections", action="store_true")
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--augment", action="store_true", help="random rotation of every training batch")
    p.add_argument("--attn_sup", type=float, default=0.0,
                   help="weight of auxiliary CE loss pushing pooled attention onto the asked atom (force questions); 0 = off")
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train_fraction", type=float, default=1.0, help="for sample-efficiency curves")
    p.add_argument("--eval_rotations", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=0, help="debug: stop early")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="default: cuda, else mps, else cpu")
    p.add_argument("--test_limit", type=int, default=0, help="debug: only evaluate this many questions per split")
    p.add_argument("--resume", action="store_true", help="resume latest completed epoch in run directory")
    p.add_argument("--output_dir", help="explicit persistent run directory")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    out_dir = args.output_dir or f"runs/{args.name}"
    os.makedirs(out_dir, exist_ok=True)
    log = open(f"{out_dir}/log.txt", "a" if args.resume else "w")

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

    if args.model == "pipeline":
        from src.baselines import PipelineForce
        data["with_atom"] = True
        model = PipelineForce().to(device)
        attach_parsed_atoms(splits, "data/gdb9.sdf", "data/qm9_xtb.pkl", "data/tensorqa_v0.json", say)
    elif args.model == "scalar":
        from src.baselines import ScalarCollapse
        model = ScalarCollapse(text_dim=data["text_dim"]).to(device)
    else:
        tiny = dict(hidden_irreps="8x0e + 2x0o + 4x1o + 2x1e + 2x2e",
                    cross_irreps="4x0e + 4x1o + 2x1e + 2x2e") if args.tiny else {}
        model = YLM(text_dim=data["text_dim"], text_mode="taskid" if args.text == "taskid" else "text",
                    option=args.option, legacy_odd_scalars=args.legacy_odd_scalars, **tiny).to(device)
    say("parameters", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps_per_epoch = math.ceil(len(train["mol"]) / args.batch_size)
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)

    step, best_val, t0 = 0, float("inf"), time.time()
    start_epoch = 0
    latest = f"{out_dir}/latest.pt"
    if args.resume and os.path.exists(latest):
        saved = torch.load(latest, map_location=device, weights_only=False)
        for key in ("epochs", "batch_size", "lr", "text", "option", "model", "tiny", "seed", "train_fraction", "augment", "legacy_odd_scalars"):
            if saved["args"][key] != vars(args)[key]:
                raise ValueError(f"Resume configuration differs: {key}")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        start_epoch, step, best_val = saved["epoch"] + 1, saved["step"], saved["best_val"]
        torch.set_rng_state(saved["rng"].cpu())
        if saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        say("resumed epoch", start_epoch, "step", step)
    profile_dir = f"{out_dir}/profiling"
    if os.path.exists(profile_dir):
        # Preserve earlier sessions and interrupted traces when resuming.
        import shutil
        shutil.move(profile_dir, f"{profile_dir}_session_{time.time_ns()}")
    profiler = Profiler(profile_dir, device=device, trace_steps=20)
    for epoch in range(start_epoch, args.epochs):
        if args.max_steps and step >= args.max_steps:
            break
        perm = torch.randperm(len(train["mol"]))
        loss_sum, n_loss = None, 0  # summed on the device: no per-step host sync
        profiler.epoch_start()
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start: start + args.batch_size]
            with profiler.phase("data"):
                R = o3.rand_matrix() if args.augment else None
                inputs, target = get_batch(data, train, idx, device, R)
            with profiler.phase("forward"):
                y, attn = model(*inputs)
                loss = ((y - target / scale_dev[inputs[5]][:, None]) ** 2).mean()
                if args.attn_sup > 0 and attn is not None:
                    loss = loss + args.attn_sup * attention_supervision_loss(
                        attn, train["atom"][idx].to(device), inputs[5], inputs[2])
            with profiler.phase("backward"):
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            with profiler.phase("optimizer"):
                opt.step()
                sched.step()
            profiler.step_done(tokens=0, samples=len(idx))
            loss_sum = loss.detach() if loss_sum is None else loss_sum + loss.detach()
            n_loss += 1
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        val = evaluate(model, data, splits["val"], scale, device, n_rot=1, batch_size=args.eval_batch_size)
        val_score = val["dipole"]["mae"] / scale[0].item() + val["force"]["mae"] / scale[1].item()
        if val_score < best_val:
            best_val = val_score
            torch.save(model.state_dict(), f"{out_dir}/model.pt.tmp")
            os.replace(f"{out_dir}/model.pt.tmp", f"{out_dir}/model.pt")
        mean_loss = (loss_sum / max(n_loss, 1)).item() if loss_sum is not None else float("nan")
        say(f"epoch {epoch} loss {mean_loss:.4f} | val dipole mae {val['dipole']['mae']:.4f} "
            f"cos {val['dipole']['cosine']:.3f} | force mae {val['force']['mae']:.5f} "
            f"cos {val['force']['cosine']:.3f} atom_acc {val['force']['atom_acc']:.3f} | {time.time() - t0:.0f}s")
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                        scheduler=sched.state_dict(), epoch=epoch, step=step, best_val=best_val,
                        args=vars(args), rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                   latest + ".tmp")
        os.replace(latest + ".tmp", latest)
        profiler.save()
        if args.max_steps and step >= args.max_steps:
            break

    profiler.close()
    model.load_state_dict(torch.load(f"{out_dir}/model.pt"))
    results = {"args": vars(args), "scale": scale.tolist()}
    if args.model == "pipeline":
        results["parser_accuracy"] = {k: splits[k]["parse_acc"] for k in ("val", "test", "test_heldout")}
    for split in ["test", "test_heldout"]:
        if args.test_limit:
            splits[split] = {k: (v[: args.test_limit] if torch.is_tensor(v) else v) for k, v in splits[split].items()}
        results[split] = evaluate(model, data, splits[split], scale, device, n_rot=args.eval_rotations, batch_size=args.eval_batch_size, routing_audit=args.routing_audit)
        if args.eval_reflections:
            results[split + "_o3"] = evaluate(model, data, splits[split], scale, device,
                n_rot=args.eval_rotations, batch_size=args.eval_batch_size, reflection=True)
            results[split + "_reflection"] = evaluate(model, data, splits[split], scale, device,
                n_rot=args.eval_rotations, batch_size=args.eval_batch_size,
                reflection=True, reflection_only=True)
        say(split, json.dumps(results[split]))
    torch.save(dict(state_dict=model.state_dict(), scale=scale.tolist(), args=vars(args),
                    text=args.text, text_dim=data["text_dim"]), f"{out_dir}/checkpoint.pt.tmp")
    os.replace(f"{out_dir}/checkpoint.pt.tmp", f"{out_dir}/checkpoint.pt")
    with open(f"{out_dir}/results.json.tmp", "w") as stream:
        json.dump(results, stream, indent=2)
    os.replace(f"{out_dir}/results.json.tmp", f"{out_dir}/results.json")
    say("saved", out_dir)


if __name__ == "__main__":
    main()

"""CPU latency (p50/p95 per request) and accuracy of src.api.YLMPredictor on TensorQA-v0.

    python scripts/bench_api.py --data data/tensorqa_v0 --n 300 --out results/cpu_api/bench.json

--data is the folder with test.jsonl / test_heldout.jsonl (the tensorqa_v0.tar.gz release asset).
Each variant answers the same seeded questions. Wall time includes waiting for a busy shared
CPU; process CPU time (1 thread) is what a request costs on an idle core. "cold" clears the question cache before every
request (text encoder always runs); "warm" keeps it. Accuracy is against the xTB labels and,
as agreement, against the full fp32 MatSciBERT checkpoint (the recommended v0 model).
"""
import argparse
import json
import os
import platform
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api import ROOT, YLMPredictor, smiles_to_mol  # noqa: E402

VARIANTS = {
    "matscibert_fp32 (full)": ("ylm_matscibert_s0.pt", False),
    "matscibert_int8": ("ylm_matscibert_s0.pt", True),
    "minilm_fp32": ("ylm_minilm_s0.pt", False),
    "minilm_int8": ("ylm_minilm_s0.pt", True),
}


def cpu_name():
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def load_rows(folder, n, seed):
    rows = []
    for split in ("test", "test_heldout"):
        with open(os.path.join(folder, f"{split}.jsonl")) as fh:
            data = [json.loads(line) for line in fh]
        rows += [dict(r, split=split) for r in random.Random(seed).sample(data, min(n, len(data)))]
    return rows


SMILES = ["CCO", "CC(=O)O", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
          "OCC(O)CO", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "NC(=O)c1ccncc1", "FC(F)(F)c1ccccc1", "C1CCOC1"]


def pct(xs, q):
    return float(np.percentile(xs, q))


def run(pred, rows, cold):
    """-> predictions, wall ms, process CPU ms (1 thread: the cost on an idle core)."""
    out, wall, cpu = [], [], []
    for r in rows:
        if cold:
            pred._cache.clear()
        t, c = time.perf_counter(), time.process_time()
        res = pred.predict(r["question"], atomic_numbers=r["atomic_numbers"], positions=r["positions_bohr"],
                           unit="bohr")
        wall.append((time.perf_counter() - t) * 1e3)
        cpu.append((time.process_time() - c) * 1e3)
        out.append(res)
    return out, wall, cpu


def stats(xs):
    return {"p50": pct(xs, 50), "p95": pct(xs, 95), "max": max(xs)}


def score(rows, preds, ref=None):
    rep = {}
    for split in ("test", "test_heldout"):
        for task in ("dipole", "force"):
            idx = [i for i, r in enumerate(rows) if r["split"] == split and r["task"] == task]
            if not idx:
                continue
            y = torch.tensor([preds[i]["vector"] for i in idx])
            t = torch.tensor([rows[i]["answer"] for i in idx])
            m = {"n": len(idx), "mae": float((y - t).abs().mean()),
                 "cos": float(torch.nn.functional.cosine_similarity(y, t, dim=-1).mean()),
                 "task_inferred_ok": float(np.mean([preds[i]["task"] == task for i in idx]))}
            if task == "force":
                m["atom_hit"] = float(np.mean([preds[i]["routed_atom"] == rows[i]["atom_index"] for i in idx]))
            if ref is not None:
                yr = torch.tensor([ref[i]["vector"] for i in idx])
                m["cos_vs_full"] = float(torch.nn.functional.cosine_similarity(y, yr, dim=-1).mean())
                m["rel_diff_vs_full"] = float(((y - yr).norm(dim=-1) / yr.norm(dim=-1).clamp_min(1e-8)).mean())
            rep[f"{split}/{task}"] = m
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "tensorqa_v0"))
    ap.add_argument("--n", type=int, default=300, help="questions per split")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = load_rows(args.data, args.n, args.seed)
    report = {"cpu": cpu_name(), "logical_cpus": os.cpu_count(), "threads": args.threads,
              "load_avg_start": os.getloadavg(), "torch": torch.__version__, "n_requests": len(rows),
              "variants": {}}
    ref = None
    for name in args.variants:
        ckpt, int8 = VARIANTS[name]
        t = time.perf_counter()
        pred = YLMPredictor(ROOT / "checkpoints" / "v0" / ckpt, threads=args.threads, int8_text=int8)
        load_s = time.perf_counter() - t
        run(pred, rows[:5], cold=True)  # warmup
        preds, cold, cold_cpu = run(pred, rows, cold=True)
        _, warm, warm_cpu = run(pred, rows, cold=False)
        smi = []
        for s in SMILES:  # uncached RDKit embedding + uncached question
            pred._cache.clear()
            smiles_to_mol.cache_clear()
            t = time.perf_counter()
            pred.predict("What is the dipole moment of this molecule?", smiles=s)
            smi.append((time.perf_counter() - t) * 1e3)
        if ref is None and name.startswith("matscibert_fp32"):
            ref = preds
        v = {"load_s": load_s,
             "cold_ms": {"wall": stats(cold), "cpu": stats(cold_cpu)},
             "warm_ms": {"wall": stats(warm), "cpu": stats(warm_cpu)},
             "smiles_first_call_ms": stats(smi),
             "accuracy": score(rows, preds, ref if ref is not preds else None)}
        report["variants"][name] = v
        print(name, json.dumps({k: v[k] for k in ("cold_ms", "warm_ms")}), flush=True)
    report["load_avg_end"] = os.getloadavg()
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()

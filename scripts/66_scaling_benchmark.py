"""Measure parameter count and forward latency scaling from N=15 to N=400 atoms across Options A and C.

Compares:
  - Option A (invariant keys x equivariant values)
  - Option C channelwise (O(C) linear Clebsch-Gordan self-product)
  - Option C fully_connected (O(C^3) cubic Clebsch-Gordan self-product)

Run:
  python scripts/66_scaling_benchmark.py --out results/scaling/results.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src  # noqa: E402, F401
from src.model import YLM  # noqa: E402


CONFIGS = [
    ("Option A", dict(option="A", c_tp_mode="fully_connected")),
    ("Option C (channelwise)", dict(option="C", c_tp_mode="channelwise")),
    ("Option C (fully_connected)", dict(option="C", c_tp_mode="fully_connected")),
]

SYSTEM_SIZES = [
    (15, "QM9 small molecule"),
    (64, "Drug-like ligand / macrocycle"),
    (180, "Li+ electrolyte solvation shell"),
    (400, "Biomolecular pocket / crystal supercell"),
]


def _make_packed_cloud(n_atoms: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    z_choices = torch.tensor([1, 6, 7, 8, 9], dtype=torch.long)
    z = z_choices[torch.randint(0, len(z_choices), (1, n_atoms), generator=g)]
    side = max(4.0, (n_atoms * 14.0) ** (1.0 / 3.0))
    pos = (torch.rand((1, n_atoms, 3), generator=g) - 0.5) * side
    mask = torch.ones((1, n_atoms), dtype=torch.bool)
    text = torch.randn((1, 16, 384), generator=g)
    tmask = torch.ones((1, 16), dtype=torch.bool)
    task = torch.tensor([1], dtype=torch.long)
    return z, pos, mask, text, tmask, task


@torch.no_grad()
def measure_latency_ms(model: torch.nn.Module, batch, warmup: int = 2, repeats: int = 5) -> float:
    for _ in range(warmup):
        model(*batch)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(*batch)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return float(times[len(times) // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scaling/results.json")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    models = {}
    summary = {"status": "MEASURED", "device": "cpu", "text_dim": 384, "models": {}, "latency_ms": []}
    for label, kw in CONFIGS:
        m = YLM(text_dim=384, **kw).eval()
        n_total = sum(p.numel() for p in m.parameters())
        n_cross = sum(p.numel() for p in m.cross.parameters())
        models[label] = m
        summary["models"][label] = {
            "total_params": int(n_total),
            "cross_attn_params": int(n_cross),
            "checkpoint_mb_fp32": round(n_total * 4 / (1024 * 1024), 2),
        }

    for n_atoms, desc in SYSTEM_SIZES:
        batch = _make_packed_cloud(n_atoms)
        row = {"n_atoms": n_atoms, "regime": desc, "median_ms": {}}
        for label, m in models.items():
            ms = measure_latency_ms(m, batch, warmup=2, repeats=args.repeats)
            row["median_ms"][label] = round(ms, 2)
        summary["latency_ms"].append(row)
        print(f"N={n_atoms:3d} ({desc}): {row['median_ms']}", flush=True)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved scaling benchmark to {out_p}")


if __name__ == "__main__":
    main()

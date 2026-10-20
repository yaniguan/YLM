"""Ceiling of any 'invariant-weights x equivariant-values' readout with ONE pointer (Option A).

Option A returns y = sum_i a_i V_i with a_i >= 0, sum_i a_i = 1 (softmax weights), so y lies in the convex hull of the
per-atom vectors. Even with PERFECT per-atom vectors V_i = F_i (the true forces), the best achievable answer is the
point of conv{F_i} closest to the target. This script computes that ceiling on TensorQA-v1 for each task:
    cosine and MAE of the closest convex combination, versus the target.
(The gate's whole-molecule sum term in our Option A adds one more unconstrained-sign scalar multiple of sum_i V_i,
 so we also report the ceiling over {a conv combination} + {c * sum_i F_i, c in R}.)

  python scripts/28_convex_ceiling.py [--split test] [--n 3000]   # needs data/tensorqa_v1.json, data/qm9_xtb.pkl
"""
import argparse
import json
import pickle
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize, nnls


def project_hull(F, y):
    """closest point to y in conv{rows of F}: min ||F^T a - y||^2, a >= 0, sum a = 1 (nnls with a heavy sum constraint row)"""
    big = 1e3
    A = np.vstack([F.T, big * np.ones((1, F.shape[0]))])
    b = np.concatenate([y, [big]])
    a, _ = nnls(A, b)
    return F.T @ a


def project_hull_plus_sum(F, y):
    """closest point to y in {conv combination} + c * sum_i F_i with free c"""
    s = F.sum(0)
    best = None
    for c in np.linspace(-2, 2, 41):
        p = project_hull(F, y - c * s) + c * s
        e = np.linalg.norm(p - y)
        if best is None or e < best[0]:
            best = (e, p)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tensorqa_v1.json")
    ap.add_argument("--labels", default="data/qm9_xtb.pkl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    labels = pickle.load(open(args.labels, "rb"))
    qs = [q for q in json.load(open(args.data)) if q["split"] == args.split]
    rng = np.random.RandomState(0)
    rng.shuffle(qs)
    stats = defaultdict(lambda: defaultdict(list))
    for q in qs[: args.n]:
        F = np.asarray(labels[q["mol"]]["forces"], float)
        y = np.asarray(q["target"], float)
        if np.linalg.norm(y) < 1e-8:
            continue
        for name, fn in (("hull", project_hull), ("hull+sum", project_hull_plus_sum)):
            p = fn(F, y)
            cos = p @ y / (np.linalg.norm(p) * np.linalg.norm(y) + 1e-12)
            stats[q["task"]][name + "_cos"].append(cos)
            stats[q["task"]][name + "_mae"].append(np.abs(p - y).mean())
        stats[q["task"]]["target_mae"].append(np.abs(y).mean())
    print(f"split {args.split}: ceiling of Option A with PERFECT per-atom forces ({sum(len(v['hull_cos']) for v in stats.values())} questions)")
    out = {}
    for t, d in stats.items():
        out[t] = {k: float(np.mean(v)) for k, v in d.items()}
        print(f"  {t:6s} hull: cos {out[t]['hull_cos']:.3f} mae {out[t]['hull_mae']:.4f} | hull+sum: cos {out[t]['hull+sum_cos']:.3f} "
              f"mae {out[t]['hull+sum_mae']:.4f} | |target| mae {out[t]['target_mae']:.4f}")
    json.dump(out, open("results_convex_ceiling.json", "w"), indent=1)


if __name__ == "__main__":
    main()

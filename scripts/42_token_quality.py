"""Token-quality test (src/token_quality.py): equal token budgets of web text, unchecked agent
records and verified agent records; train a tiny GPT on each; score on a held-out,
checker-scored eval. CPU only.

Run (about 1 hour on 7 CPU cores):
  python scripts/42_token_quality.py --lablean_dir lean --web_dir WLP-Dataset --out runs/token_quality
"""
import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import lab_world as W  # noqa: E402
from src import token_quality as TQ  # noqa: E402

ARMS = ("web", "agent", "verified")


def job(args):
    import torch
    torch.set_num_threads(1)
    arm, budget, seed, text, items, steps = args
    t0 = time.time()
    if arm == "untrained":
        model, losses = TQ.untrained(seed), []
    else:
        model, losses = TQ.train(text, seed, steps=steps)
    cal, raw = TQ.evaluate(model, items)
    return {"arm": arm, "budget": budget, "seed": seed, "cal": cal.tolist(), "raw": raw.tolist(),
            "final_loss": float(np.mean(losses[-20:])) if losses else None,
            "seconds": round(time.time() - t0, 1)}


def t_interval(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return [float("nan")] * 2
    from scipy import stats
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean() - h), float(d.mean() + h)]


def boot(a, b, rng, n=10000):
    """Paired two-level bootstrap of mean(a) - mean(b); a, b are seeds x items 0/1 arrays."""
    S, I = a.shape
    out = np.empty(n)
    for k in range(n):
        si = rng.integers(0, S, S)
        ii = rng.integers(0, I, I)
        out[k] = a[np.ix_(si, ii)].mean() - b[np.ix_(si, ii)].mean()
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lablean_dir", default="lean")
    ap.add_argument("--web_dir", required=True, help="Wet Lab Protocols corpus (github.com/chaitanya2334/WLP-Dataset)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[250_000, 1_000_000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--oracle_seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--agent_tasks", type=int, default=4500)
    ap.add_argument("--eval_protocols", type=int, default=400)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", default="runs/token_quality")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    web = TQ.web_docs(a.web_dir)
    recs, used_lean = TQ.agent_records(seed=100, n_tasks=a.agent_tasks, lablean_dir=a.lablean_dir)
    train_hashes = {TQ.protocol_hash(r["task"]["state"], r["steps"]) for r in recs}
    exclude = set(train_hashes)
    oracle = TQ.oracle_docs(200, 3000, a.lablean_dir, exclude)      # adds its protocols to `exclude`
    seen_before = set(exclude)
    items = TQ.build_eval(300, a.eval_protocols, a.lablean_dir, exclude)
    eval_hashes = exclude - seen_before
    assert len(eval_hashes) == a.eval_protocols and not eval_hashes & seen_before   # eval is held out
    print(f"web docs {len(web)} ({sum(len(d) + 1 for d in web)} chars); agent records {len(recs)} "
          f"(verified {sum(r['trust'] == 'verified' for r in recs)}); eval items {len(items)}; "
          f"Lean used: {used_lean}", flush=True)

    pools = {"web": [(d, None) for d in web],
             "agent": [(r["text"], r["trust"]) for r in recs],
             "verified": [(r["text"], r["trust"]) for r in recs if r["trust"] == "verified"],
             "oracle": [(d, None) for d in oracle]}
    texts, card = {}, {}
    for arm in ARMS + ("oracle",):
        for budget in a.budgets:
            if arm == "oracle" and budget != max(a.budgets):
                continue
            rng = random.Random(budget)
            docs = list(pools[arm])
            rng.shuffle(docs)
            used, n = [], 0
            for d, trust in docs:
                used.append((d, trust))
                n += len(d) + 1
                if n >= budget:
                    break
            text = "\n".join(d for d, _ in used)[:budget]
            assert len(text) == budget, (arm, budget, len(text))
            texts[(arm, budget)] = text
            trusts = [t for _, t in used if t is not None]
            card[f"{arm}@{budget}"] = {
                "docs": len(used), "chars": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "share_rejected_by_gate": (sum(t != "verified" for t in trusts) / len(trusts))
                if trusts else None}
    print(json.dumps(card, indent=1), flush=True)

    jobs = [(arm, b, s, texts[(arm, b)], items, a.steps) for b in a.budgets for arm in ARMS for s in a.seeds]
    jobs += [("oracle", max(a.budgets), s, texts[("oracle", max(a.budgets))], items, a.steps) for s in a.oracle_seeds]
    jobs += [("untrained", 0, s, None, items, 0) for s in a.seeds]
    results = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(job, jobs):
            results.append(r)
            print(f"{r['arm']:>9} D={r['budget']:>8} seed {r['seed']}: acc {np.mean(r['cal']):.3f} "
                  f"(raw {np.mean(r['raw']):.3f}) loss {r['final_loss']} {r['seconds']}s", flush=True)
            with open(os.path.join(a.out, "runs.jsonl"), "a") as f:
                f.write(json.dumps(r) + "\n")

    kinds = np.array([it["kind"] for it in items])

    def mat(arm, budget):
        rows = sorted((r for r in results if r["arm"] == arm and r["budget"] == budget), key=lambda r: r["seed"])
        return np.array([r["cal"] for r in rows]), np.array([r["raw"] for r in rows]), [r["seed"] for r in rows]

    base_cal, _, _ = mat("untrained", 0)
    base = float(base_cal.mean())
    table = {}
    for arm in ARMS + ("oracle",):
        for b in a.budgets:
            cal, raw, seeds = mat(arm, b)
            if not len(cal):
                continue
            per_seed = cal.mean(1)
            table[f"{arm}@{b}"] = {
                "seeds": seeds, "acc_mean": float(per_seed.mean()), "acc_sd": float(per_seed.std(ddof=1)) if len(seeds) > 1 else None,
                "acc_per_seed": per_seed.tolist(), "raw_acc_mean": float(raw.mean()),
                "acc_by_kind": {k: float(cal[:, kinds == k].mean()) for k in sorted(set(kinds))},
                "gain_vs_untrained": float(per_seed.mean() - base),
                "gain_per_1M_tokens": float((per_seed.mean() - base) / (b / 1e6))}
    rng = np.random.default_rng(0)
    diffs = {}
    for b in a.budgets:
        for x, y in (("verified", "agent"), ("verified", "web"), ("agent", "web")):
            A, _, sa = mat(x, b)
            B, _, sb = mat(y, b)
            assert sa == sb
            d = A.mean(1) - B.mean(1)
            diffs[f"{x}-{y}@{b}"] = {"mean": float(d.mean()), "per_seed": d.tolist(),
                                     "t95": t_interval(d), "boot95": boot(A, B, rng)}
    out = {"table": table, "untrained_acc": base, "chance": 0.25, "differences": diffs, "data_card": card,
           "eval": {"items": len(items), "by_kind": {k: int((kinds == k).sum()) for k in sorted(set(kinds))},
                    "scored_by": "LabLean" if used_lean else "Python copy of LabLean"},
           "args": vars(a), "seconds": round(time.time() - t0, 1),
           "machine": f"{platform.system()} {platform.machine()}, CPU only, {a.workers} worker processes"}
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nuntrained acc {base:.3f} (chance 0.25)")
    for k, v in table.items():
        print(f"{k:>18}: acc {v['acc_mean']:.3f} +- {v['acc_sd'] if v['acc_sd'] is None else round(v['acc_sd'], 3)}"
              f"  verdict {v['acc_by_kind'].get('verdict', 0):.3f}  volume {v['acc_by_kind'].get('volume', 0):.3f}"
              f"  gain/1M tok {v['gain_per_1M_tokens']:+.3f}")
    for k, v in diffs.items():
        print(f"{k:>24}: {v['mean']:+.3f}  t95 [{v['t95'][0]:+.3f}, {v['t95'][1]:+.3f}]  "
              f"boot95 [{v['boot95'][0]:+.3f}, {v['boot95'][1]:+.3f}]")


if __name__ == "__main__":
    main()

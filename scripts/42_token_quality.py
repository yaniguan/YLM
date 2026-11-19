"""Token-quality test (src/token_quality.py): equal token budgets of web text, unchecked agent
records and verified agent records; train a tiny GPT on each; score on a held-out,
checker-scored eval. CPU only.

Scoring rules (declared in docs/TOKEN_QUALITY.md before any re-scoring):
  primary   unpadded likelihood, argmax sum log p(option | question), on eval v2
  gate      the oracle beats the untrained model on the primary score, AND the paired
            bootstrap 95% CI of (oracle - oracle with questions shuffled within level) lies
            above 0. If the gate fails, nothing is concluded about the arms.
  secondary always reported: padded raw, padded calibrated, per-character, newline-terminated,
            and the prior-only (question-free) baseline.

Runs:
  # gate only (oracle seeds + untrained); run this first and stop if it fails
  python scripts/42_token_quality.py --lablean_dir lean --gate_only
  # all arms
  python scripts/42_token_quality.py --lablean_dir lean --web_dir WLP-Dataset
  # data card and eval statistics only, no training
  python scripts/42_token_quality.py --lablean_dir lean --data_card_only --arms agent verified oracle

Memory: each training worker peaks at about GB_PER_WORKER GB, so by default the number of
worker processes is MemAvailable / GB_PER_WORKER (at least 1, at most the CPU count and the
number of jobs). Override with --workers.

Outputs (in --out, which defaults to a fresh runs/token_quality/<date-time> directory):
  data_card.json  training texts (sizes, SHA-256, rejected share and reasons) and eval statistics
  eval_items.json the eval v2 items (question, options, answer), to re-score saved log-probs
  runs.jsonl      one row per run: per-item correctness for every score, per-option log-probs,
                  and eval v1 correctness. Created fresh; the script refuses to append to an old one.
  results.json    tables, the gate, paired differences
"""
import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import token_quality as TQ  # noqa: E402

ARMS = ("web", "agent", "verified")
ALL_ARMS = ARMS + ("oracle",)
GB_PER_WORKER = 0.93          # peak memory of one training worker (1 thread, CTX 512, batch 8)
REASONS = {"V0_format": "bad format", "V1_lablean": "wrong verdict (LabLean)",
           "V1b_mirror": "wrong verdict (Python copy of LabLean)",
           "V2_mirror_agrees": "Python copy disagrees with LabLean", "V3_volumes": "wrong volumes",
           "dedup": "exact duplicate"}


def mem_available_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


def default_workers(n_jobs, gb_per_worker=GB_PER_WORKER):
    """Workers that fit in free RAM: MemAvailable / gb_per_worker, within [1, CPUs, jobs]."""
    gb = mem_available_gb()
    fit = int(gb // gb_per_worker) if gb is not None else 1
    return max(1, min(fit, os.cpu_count() or 1, n_jobs))


def neutral_path(p):
    """A path for the output files: relative to the repo if inside it, else its last part."""
    if p is None:
        return None
    ap = os.path.abspath(p)
    if ap == ROOT or ap.startswith(ROOT + os.sep):
        return os.path.relpath(ap, ROOT)
    return os.path.basename(os.path.normpath(p))


def build_texts(pools, budgets, arms):
    """Cut each arm to exactly D characters with TQ.stream (oracle at the largest D only).

    Returns {(arm, D): (text, ids)}, where ids index the documents of pools[arm] that were used.
    """
    out = {}
    for arm in arms:
        for budget in budgets:
            if arm == "oracle" and budget != max(budgets):
                continue
            text, ids = TQ.stream([d for d, _ in pools[arm]], budget, random.Random(budget), return_ids=True)
            assert len(text) == budget, (arm, budget, len(text))
            out[(arm, budget)] = (text, ids)
    return out


def reason_counts(recs):
    rej = [r for r in recs if r["trust"] != "verified"]
    return dict(sorted(Counter(r["first_fail"] for r in rej).items()))


def job(args):
    import torch
    torch.set_num_threads(1)
    arm, budget, seed, text, items, items_v1, steps = args
    t0 = time.time()
    if arm == "untrained":
        model, losses = TQ.untrained(seed), []
    else:
        model, losses = TQ.train(text, seed, steps=steps)
    lps = TQ.evaluate(model, items)
    corr = TQ.correct(lps, items)
    cal, raw = TQ.evaluate_v1(model, items_v1)
    return {"arm": arm, "budget": budget, "seed": seed,
            "correct": {k: v.tolist() for k, v in corr.items()},
            "logprobs": {k: np.round(v, 4).tolist() for k, v in lps.items()},
            "v1": {"cal": cal.tolist(), "raw": raw.tolist()},
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


def gate(results, oracle_budget, rng):
    """The positive-control rule declared in docs/TOKEN_QUALITY.md."""
    O, so = mat(results, "oracle", oracle_budget, "primary")
    S, _ = mat(results, "oracle", oracle_budget, "shuffled_q")
    U, su = mat(results, "untrained", 0, "primary")
    if not len(O) or not len(U):
        return None
    beats = bool(O.mean() > U.mean())
    ci = boot(O, S, rng)
    return {"rule": "oracle primary acc > untrained primary acc, and the bootstrap 95% CI of "
                    "(oracle - oracle with questions shuffled within level) lies above 0",
            "oracle_seeds": so, "untrained_seeds": su,
            "oracle_primary": float(O.mean()), "untrained_primary": float(U.mean()),
            "oracle_beats_untrained": beats,
            "oracle_shuffled_q": float(S.mean()), "oracle_minus_shuffled": float(O.mean() - S.mean()),
            "oracle_minus_shuffled_boot95": ci,
            "passed": bool(beats and ci[0] > 0)}


def mat(results, arm, budget, score):
    rows = sorted((r for r in results if r["arm"] == arm and r["budget"] == budget), key=lambda r: r["seed"])
    return np.array([r["correct"][score] for r in rows]), [r["seed"] for r in rows]


def compare_v1(results, path):
    """Item-for-item comparison of eval v1 correctness with an earlier runs.jsonl."""
    old = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            old[(r["arm"], r["budget"], r["seed"])] = r
    same, compared, wrong_size = [], 0, 0
    for r in results:
        o = old.get((r["arm"], r["budget"], r["seed"]))
        if o is None:
            continue
        if len(o["cal"]) != len(r["v1"]["cal"]):
            wrong_size += 1                    # a different eval v1 (other --eval_protocols)
            continue
        compared += 1
        if o["cal"] == r["v1"]["cal"] and o["raw"] == r["v1"]["raw"]:
            same.append(f"{r['arm']}@{r['budget']} seed {r['seed']}")
    return {"compared_with": neutral_path(path), "runs_compared": compared, "runs_identical": len(same),
            "identical": sorted(same), "runs_with_a_different_item_count": wrong_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lablean_dir", default="lean")
    ap.add_argument("--web_dir", help="Wet Lab Protocols corpus (github.com/chaitanya2334/WLP-Dataset); "
                                      "needed for the web arm")
    ap.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(ALL_ARMS))
    ap.add_argument("--gate_only", action="store_true",
                    help="run only the positive-control gate: oracle seeds + untrained (same as --arms oracle)")
    ap.add_argument("--data_card_only", action="store_true",
                    help="write data_card.json and eval_items.json, no training")
    ap.add_argument("--budgets", type=int, nargs="+", default=[250_000, 1_000_000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--oracle_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--agent_tasks", type=int, default=4500)
    ap.add_argument("--eval_protocols", type=int, default=400)
    ap.add_argument("--workers", type=int, default=None,
                    help=f"worker processes (default: free RAM / {GB_PER_WORKER} GB, at most the CPU count)")
    ap.add_argument("--compare_v1", default=None,
                    help="an earlier runs.jsonl (e.g. results/token_quality/runs.jsonl) to compare eval v1 "
                         "correctness with, item for item")
    ap.add_argument("--out", default=None, help="output directory (default: runs/token_quality/<date-time>)")
    a = ap.parse_args()
    arms = ("oracle",) if a.gate_only else tuple(x for x in ALL_ARMS if x in a.arms)
    if "web" in arms and not a.web_dir:
        ap.error("--web_dir is needed for the web arm (or leave it out with --arms)")
    a.out = a.out or os.path.join("runs", "token_quality", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(a.out, exist_ok=True)
    runs_path = os.path.join(a.out, "runs.jsonl")
    if not a.data_card_only and os.path.exists(runs_path):
        sys.exit(f"{runs_path} exists; use a new --out so runs from different invocations never mix")
    t0 = time.time()

    web = TQ.web_docs(a.web_dir) if "web" in arms else []
    recs, used_lean = TQ.agent_records(seed=100, n_tasks=a.agent_tasks, lablean_dir=a.lablean_dir)
    train_hashes = {TQ.protocol_hash(r["task"]["state"], r["steps"]) for r in recs}
    exclude = set(train_hashes)
    oracle = TQ.oracle_docs(200, 3000, a.lablean_dir, exclude)      # adds its protocols to `exclude`
    seen_before = set(exclude)
    items_v1 = TQ.build_eval_v1(300, a.eval_protocols, a.lablean_dir, set(exclude))
    eval_stats = {}
    items = TQ.build_eval(300, a.eval_protocols, a.lablean_dir, exclude, stats=eval_stats)
    eval_hashes = exclude - seen_before
    assert len(eval_hashes) == a.eval_protocols and not eval_hashes & seen_before   # eval is held out
    print(f"web docs {len(web)}; agent records {len(recs)} (verified {sum(r['trust'] == 'verified' for r in recs)}); "
          f"eval items {len(items)} (v1: {len(items_v1)}); Lean used: {used_lean}", flush=True)

    pools = {"web": [(d, None) for d in web],
             "agent": [(r["text"], r) for r in recs],
             "verified": [(r["text"], r) for r in recs if r["trust"] == "verified"],
             "oracle": [(d, None) for d in oracle]}
    texts = build_texts(pools, a.budgets, tuple(dict.fromkeys(arms + ("oracle",))))
    checked_by = "LabLean" if used_lean else "Python copy of LabLean"
    card = {"agent_pool": {"records": len(recs), "verified": sum(r["trust"] == "verified" for r in recs),
                           "share_rejected_by_gate": sum(r["trust"] != "verified" for r in recs) / len(recs),
                           "rejected_by_reason": reason_counts(recs), "checked_by": checked_by},
            "reason_names": REASONS}
    for (arm, budget), (text, ids) in texts.items():
        used = [pools[arm][i][1] for i in ids]
        recs_used = [r for r in used if r is not None]
        card[f"{arm}@{budget}"] = {
            "docs": len(ids), "chars": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "share_rejected_by_gate": (sum(r["trust"] != "verified" for r in recs_used) / len(recs_used))
            if recs_used else None,
            "rejected_by_reason": reason_counts(recs_used) if recs_used else None}
    oracle_budget = max(a.budgets)
    prior_docs = [oracle[i] for i in texts[("oracle", oracle_budget)][1]]
    kinds, kinds_v1 = np.array([it["kind"] for it in items]), np.array([it["kind"] for it in items_v1])
    p2, p1 = TQ.prior_only_correct(items, prior_docs), TQ.prior_only_correct(items_v1, prior_docs)
    card["eval"] = {"version": 2, "items": len(items),
                    "items_sha256": hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest(),
                    "by_kind": {k: int((kinds == k).sum()) for k in sorted(set(kinds))},
                    "scored_by": checked_by, "construction": eval_stats,
                    "prior_only_acc": float(p2.mean()),
                    "prior_only_by_kind": {k: float(p2[kinds == k].mean()) for k in sorted(set(kinds))},
                    "prior_docs": f"the {len(prior_docs)} documents of oracle@{oracle_budget}"}
    card["eval_v1"] = {"items": len(items_v1),
                       "by_kind": {k: int((kinds_v1 == k).sum()) for k in sorted(set(kinds_v1))},
                       "truncated_items": sum(len(it["q"]) + max(len(o) for o in it["options"]) > TQ.CTX + 1
                                              for it in items_v1),
                       "prior_only_acc": float(p1.mean()),
                       "prior_only_by_kind": {k: float(p1[kinds_v1 == k].mean()) for k in sorted(set(kinds_v1))}}
    with open(os.path.join(a.out, "data_card.json"), "w") as f:
        json.dump(card, f, indent=1)
    with open(os.path.join(a.out, "eval_items.json"), "w") as f:
        json.dump(items, f, indent=0)
    print(json.dumps(card, indent=1), flush=True)
    if a.data_card_only:
        return

    train_arms = tuple(x for x in arms if x != "oracle")
    jobs = [(arm, b, s, texts[(arm, b)][0], items, items_v1, a.steps)
            for b in a.budgets for arm in train_arms for s in a.seeds]
    if "oracle" in arms:
        jobs += [("oracle", oracle_budget, s, texts[("oracle", oracle_budget)][0], items, items_v1, a.steps)
                 for s in a.oracle_seeds]
    jobs += [("untrained", 0, s, None, items, items_v1, 0) for s in a.seeds]
    workers = a.workers or default_workers(len(jobs))
    print(f"{len(jobs)} jobs on {workers} worker processes", flush=True)
    results = []
    with open(runs_path, "x") as runs_file, Pool(workers) as pool:
        for r in pool.imap_unordered(job, jobs):
            results.append(r)
            c = r["correct"]
            print(f"{r['arm']:>9} D={r['budget']:>8} seed {r['seed']}: primary {np.mean(c['primary']):.3f} "
                  f"shuffled-q {np.mean(c['shuffled_q']):.3f} padded raw {np.mean(c['padded_raw']):.3f} "
                  f"padded cal {np.mean(c['padded_calibrated']):.3f} loss {r['final_loss']} {r['seconds']}s",
                  flush=True)
            runs_file.write(json.dumps(r) + "\n")
            runs_file.flush()

    base = {s: float(mat(results, "untrained", 0, s)[0].mean()) for s in TQ.SCORES}
    table = {}
    for arm in ALL_ARMS:
        for b in a.budgets:
            prim, seeds = mat(results, arm, b, "primary")
            if not len(prim):
                continue
            scores = {}
            for s in TQ.SCORES:
                m, _ = mat(results, arm, b, s)
                per_seed = m.mean(1)
                scores[s] = {"acc_mean": float(per_seed.mean()), "acc_per_seed": per_seed.tolist(),
                             "acc_sd": float(per_seed.std(ddof=1)) if len(seeds) > 1 else None}
            rows = [r for r in results if r["arm"] == arm and r["budget"] == b]
            table[f"{arm}@{b}"] = {
                "seeds": seeds, "scores": scores,
                "primary_by_kind": {k: float(prim[:, kinds == k].mean()) for k in sorted(set(kinds))},
                "primary_gain_vs_untrained": float(prim.mean() - base["primary"]),
                "v1_calibrated_acc": float(np.mean([r["v1"]["cal"] for r in rows])),
                "v1_raw_acc": float(np.mean([r["v1"]["raw"] for r in rows]))}
    rng = np.random.default_rng(0)
    diffs = {}
    for b in a.budgets:
        for x, y in (("verified", "agent"), ("verified", "web"), ("agent", "web")):
            if x not in arms or y not in arms:
                continue
            A, sa = mat(results, x, b, "primary")
            B, sb = mat(results, y, b, "primary")
            assert sa == sb
            d = A.mean(1) - B.mean(1)
            diffs[f"{x}-{y}@{b}"] = {"mean": float(d.mean()), "per_seed": d.tolist(),
                                     "t95": t_interval(d), "boot95": boot(A, B, rng)}
    g = gate(results, oracle_budget, np.random.default_rng(1)) if "oracle" in arms else None
    args = dict(vars(a), arms=list(arms), lablean_dir=neutral_path(a.lablean_dir), web_dir=neutral_path(a.web_dir),
                out=neutral_path(a.out), compare_v1=neutral_path(a.compare_v1), workers=workers)
    out = {"primary_score": "unpadded likelihood: argmax sum log p(option | question)",
           "gate": g, "table": table, "untrained": base, "chance": 0.25,
           "prior_only": {"eval_v2": card["eval"]["prior_only_acc"], "eval_v1": card["eval_v1"]["prior_only_acc"]},
           "differences": diffs, "data_card": card,
           "eval": {k: card["eval"][k] for k in ("version", "items", "items_sha256", "by_kind", "scored_by",
                                                 "construction")},
           "v1_comparison": compare_v1(results, a.compare_v1) if a.compare_v1 else None,
           "args": args, "seconds": round(time.time() - t0, 1),
           "machine": f"{platform.system()} {platform.machine()}, CPU only, {workers} worker processes"}
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nuntrained primary {base['primary']:.3f}; prior-only {out['prior_only']['eval_v2']:.3f} (chance 0.25)")
    for k, v in table.items():
        sc = v["scores"]
        sd = sc["primary"]["acc_sd"]
        print(f"{k:>18}: primary {sc['primary']['acc_mean']:.3f} +- {sd if sd is None else round(sd, 3)}"
              f"  shuffled-q {sc['shuffled_q']['acc_mean']:.3f}  per-char {sc['per_char']['acc_mean']:.3f}"
              f"  newline {sc['newline']['acc_mean']:.3f}  padded raw {sc['padded_raw']['acc_mean']:.3f}"
              f"  padded cal {sc['padded_calibrated']['acc_mean']:.3f}")
    for k, v in diffs.items():
        print(f"{k:>24}: {v['mean']:+.3f}  t95 [{v['t95'][0]:+.3f}, {v['t95'][1]:+.3f}]  "
              f"boot95 [{v['boot95'][0]:+.3f}, {v['boot95'][1]:+.3f}]")
    if g is not None:
        print(f"gate: oracle {g['oracle_primary']:.3f} vs untrained {g['untrained_primary']:.3f}; "
              f"oracle - shuffled {g['oracle_minus_shuffled']:+.3f} boot95 "
              f"[{g['oracle_minus_shuffled_boot95'][0]:+.3f}, {g['oracle_minus_shuffled_boot95'][1]:+.3f}] -> "
              f"{'PASSED' if g['passed'] else 'FAILED (conclude nothing about the arms)'}")


if __name__ == "__main__":
    main()

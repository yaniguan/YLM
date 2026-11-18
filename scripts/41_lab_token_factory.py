"""Run the lab token factory (src/lab_token_factory.py) and write kept / rejected records.

Needs a built LabLean checkout (YLM PR #92, `lean/` after `lake build`) for the Lean gate.
Without it the Python copy of LabLean is used alone and records say V1_lablean = "skipped".

Outputs in --out:
  kept.jsonl      verified records: {"text", "provenance", ...}
  rejected.jsonl  rejected records with the first failing checker in provenance.first_fail
  stats.json      per-round yields, trusted levels, agent blind spots, checker versions, timings,
                  and how much wrong SQL passes the gate by luck with one / two databases

Run:
  python scripts/41_lab_token_factory.py --lablean_dir lean --seeds 0 1 2 --out runs/lab_tf
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import lab_world as W  # noqa: E402
from src.lab_token_factory import FACTORY_VERSION, run_factory  # noqa: E402
from src.sql_gate import SQL_GATE_VERSION, lucky_pass_counts  # noqa: E402


def lablean_version(d):
    if not W.lean_available(d):
        return None
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d, capture_output=True, text=True).stdout.strip()
    tool = open(os.path.join(d, "lean-toolchain")).read().strip()
    return {"git": sha or "unknown", "toolchain": tool}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lablean_dir", default="lean")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--per_round", type=int, default=200)
    ap.add_argument("--sql_per_round", type=int, default=50)
    ap.add_argument("--no_learn", action="store_true", help="agent does not learn from refuted claims")
    ap.add_argument("--sql_lucky_seeds", type=int, default=200,
                    help="databases used to count wrong SQL that passes the gate by luck (0 = skip)")
    ap.add_argument("--out", default="runs/lab_tf")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    versions = {"lablean": lablean_version(a.lablean_dir), "mirror": "lab_world-v1",
                "sql_gate": SQL_GATE_VERSION}
    t0 = time.time()
    kept, rejected, per_seed = [], [], []
    seen = set()                                     # dedup across seeds, not only within one
    for seed in a.seeds:
        k, r, s = run_factory(seed=seed, rounds=a.rounds, per_round=a.per_round,
                              sql_per_round=a.sql_per_round, lablean_dir=a.lablean_dir,
                              learn=not a.no_learn, checker_versions=versions, seen=seen)
        kept += k
        rejected += r
        per_seed.append(s)
        for x in s["rounds"]:
            print(f"seed {seed} round {x['round']}: trusted levels 1-{x['frontier']} -> 1-{x['frontier_after']}"
                  f"  blind spots {x['ignore_before']}  protocol kept {x['protocol_kept']}/{x['protocol_total']}"
                  f"  sql kept {x['sql_kept']}/{x['sql_total']} (wrong SQL that passed: {x['sql_wrong_but_passed']})")
    with open(os.path.join(a.out, "kept.jsonl"), "w") as f:
        for x in kept:
            f.write(json.dumps(x) + "\n")
    with open(os.path.join(a.out, "rejected.jsonl"), "w") as f:
        for x in rejected:
            f.write(json.dumps(x) + "\n")
    factory_seconds = round(time.time() - t0, 1)
    allr = kept + rejected
    lean_checked = sum(x["provenance"]["verdicts"].get("V1_lablean") in ("pass", "fail") for x in allr)
    sql_lucky = None
    if a.sql_lucky_seeds:
        sql_lucky = {"seeds": a.sql_lucky_seeds, **lucky_pass_counts(range(a.sql_lucky_seeds))}
    stats = {
        "factory": FACTORY_VERSION, "checkers": versions, "args": vars(a),
        "kept": len(kept), "rejected": len(rejected),
        "kept_by_domain": dict(Counter(x["provenance"]["domain"] for x in kept)),
        "total_by_domain": dict(Counter(x["provenance"]["domain"] for x in allr)),
        "first_fail": dict(Counter(x["provenance"]["first_fail"] for x in rejected)),
        # null when Lean never ran: V2 was skipped, so 0 would mean nothing
        "mirror_disagrees_with_lean": sum(x["provenance"]["verdicts"].get("V2_mirror_agrees") == "fail"
                                          for x in allr) if lean_checked else None,
        "lean_checked": lean_checked,
        "kept_chars": sum(len(x["text"]) for x in kept),
        "sql_lucky_pass": sql_lucky,
        "per_seed": per_seed,
        "seconds": factory_seconds,
        "machine": f"{platform.system()} {platform.machine()}, CPU only",
    }
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: stats[k] for k in ("kept", "rejected", "kept_by_domain", "total_by_domain",
                                            "first_fail", "mirror_disagrees_with_lean", "lean_checked",
                                            "sql_lucky_pass", "seconds")}, indent=1))


if __name__ == "__main__":
    main()

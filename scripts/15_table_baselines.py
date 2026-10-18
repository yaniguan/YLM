"""Post-v0 baselines and compositional results -> results_baselines.md (reads runs_hoffman2/*/results.json).

  python scripts/15_table_baselines.py
"""
import json
import os

import numpy as np

R = "runs_hoffman2"


def load(names):
    return [json.load(open(f"{R}/{n}/results.json")) for n in names if os.path.exists(f"{R}/{n}/results.json")]


def ms(v):
    return f"{np.mean(v):.4f} ± {np.std(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"


out = ["# Post-v0 baselines (not part of the v0.1.0 release)", "",
       "Mean ± std over seeds (n = number of runs). MAE: force in hartree/bohr, dipole in e·bohr.", "",
       "## v0 questions (test)", "",
       "| Model | n | force MAE | force cos | dipole cos | rotation error (mean) |", "|---|---|---|---|---|---|"]
rows = [("Pipeline: rule parser -> atom index -> equivariant force model", [f"pipe_s{s}" for s in range(3)]),
        ("Scalar collapse (invariants only, rotation-augmented)", [f"scalar_s{s}" for s in range(3)]),
        ("Qwen2.5-0.5B coordinates as text, no hint, no aug (v1)", ["llm05_noaug"]),
        ("Qwen2.5-0.5B coordinates as text, no hint, aug (v1)", ["llm05_aug"]),
        ("Qwen2.5-0.5B, no hint, lr 5e-5, 5 epochs (v2)", ["llm06_long_noaug"]),
        ("Qwen2.5-0.5B, atom hint, no aug (v2)", ["llm06_hint_noaug"]),
        ("Qwen2.5-0.5B, atom hint, aug (v2)", ["llm06_hint_aug"]),
        ("Qwen2.5-0.5B, atom hint, lr 2e-5 / head 1e-4 (v3)", ["llm07_hint_noaug"]),
        ("Qwen2.5-0.5B, no hint, lr 2e-5 / head 1e-4 (v3)", ["llm07_nohint_noaug"])]
for label, names in rows:
    rs = load(names)
    if not rs:
        out.append(f"| {label} | 0 | pending | | | |")
        continue
    t = [r["test"] for r in rs]
    eq = [x["force"]["eq_err_mean"] for x in t if x["force"]["eq_err_mean"] is not None]
    out.append(f"| {label} | {len(rs)} | {ms([x['force']['mae'] for x in t])} | {np.mean([x['force']['cosine'] for x in t]):.3f} | "
               f"{np.mean([x['dipole']['cosine'] for x in t]):.3f} | {np.mean(eq):.1e} |")
pa = load([f"pipe_s{s}" for s in range(3)])
if pa:
    out += ["", f"Parser accuracy (force questions): {pa[0]['parser_accuracy']}", ""]
out += ["", "## Compositional questions (T4 diff = F_A - F_B, T5 proj = part of F_A along A->B)", "",
        "| Model | split | n | T4 cos | T4 MAE | T5 cos | T5 MAE |", "|---|---|---|---|---|---|---|"]
for label, m in [("Option B/C-lite", "bc"), ("Option A", "optA")]:
    rs = load([f"comp_{m}_s{s}" for s in range(3)])
    for sp in ["test", "test_heldout"]:
        if not rs:
            continue
        c = lambda t, k: [r[sp][t][k] for r in rs]  # noqa: E731
        out.append(f"| {label} | {sp} | {len(rs)} | {ms(c('diff', 'cosine'))} | {ms(c('diff', 'mae'))} | "
                   f"{ms(c('proj', 'cosine'))} | {ms(c('proj', 'mae'))} |")
open("results_baselines.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))

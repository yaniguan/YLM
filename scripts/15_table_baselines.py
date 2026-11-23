"""Post-v0 baselines and compositional results -> results_baselines.md (reads runs_uge/*/results.json).

  python scripts/15_table_baselines.py
"""
import json
import os

import numpy as np

R = "results/cluster_logs/runs_uge" if os.path.exists("results/cluster_logs/runs_uge") else "runs_uge"


def load(names):
    return [json.load(open(f"{R}/{n}/results.json", encoding="utf-8")) for n in names if os.path.exists(f"{R}/{n}/results.json")]


def ms(v):
    return f"{np.mean(v):.4f} ± {np.std(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"


def main():
    out = [
        "# Post-v0 baselines (not part of the v0.1.0 release)",
        "",
        "Mean ± std over seeds (n = number of runs). MAE: force in hartree/bohr, dipole in e·bohr.",
        "",
        "## v0 questions (test)",
        "",
        "| Model | n | force MAE | force cos | dipole cos | rotation error (mean) |",
        "|---|---|---|---|---|---|",
    ]
    rows = [
        ("Pipeline: rule parser -> atom index -> equivariant force model", [f"pipe_s{s}" for s in range(3)]),
        ("Scalar collapse (invariants only, rotation-augmented)", [f"scalar_s{s}" for s in range(3)]),
        ("Qwen2.5-0.5B coordinates as text, no hint, no aug (v1)", ["llm05_noaug"]),
        ("Qwen2.5-0.5B coordinates as text, no hint, aug (v1)", ["llm05_aug"]),
        ("Qwen2.5-0.5B, no hint, lr 5e-5, 5 epochs (v2)", ["llm06_long_noaug"]),
        ("Qwen2.5-0.5B, atom hint, no aug (v2)", ["llm06_hint_noaug"]),
        ("Qwen2.5-0.5B, atom hint, aug (v2)", ["llm06_hint_aug"]),
        ("Qwen2.5-0.5B, atom hint, lr 2e-5 / head 1e-4 (v3)", ["llm07_hint_noaug"]),
        ("Qwen2.5-0.5B, no hint, lr 2e-5 / head 1e-4 (v3)", ["llm07_nohint_noaug"]),
    ]
    for label, names in rows:
        rs = load(names)
        if not rs:
            out.append(f"| {label} | 0 | not run (queued multi-GPU; see Issues #56, #58) | - | - | - |")
            continue
        t = [r["test"] for r in rs]
        eq = [x["force"]["eq_err_mean"] for x in t if x["force"]["eq_err_mean"] is not None]
        out.append(
            f"| {label} | {len(rs)} | {ms([x['force']['mae'] for x in t])} | "
            f"{np.mean([x['force']['cosine'] for x in t]):.3f} | "
            f"{np.mean([x['dipole']['cosine'] for x in t]):.3f} | {np.mean(eq):.1e} |"
        )
    pa = load([f"pipe_s{s}" for s in range(3)])
    if pa:
        out += ["", f"Parser accuracy (force questions): {pa[0]['parser_accuracy']}", ""]
    out += [
        "",
        "## Compositional questions (T4 diff = F_A - F_B, T5 proj = signed projection of F_A along A->B)",
        "",
        "| Model | split | n | T4 cos | T4 MAE | T5 cos | T5 MAE |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, m in [("Option B/C-lite", "bc"), ("Option A", "optA")]:
        rs = load([f"comp_{m}_s{s}" for s in range(3)])
        for sp in ["test", "test_heldout"]:
            if not rs:
                continue
            c = lambda t, k: [r[sp][t][k] for r in rs]  # noqa: E731
            out.append(
                f"| {label} | {sp} | {len(rs)} | {ms(c('diff', 'cosine'))} | {ms(c('diff', 'mae'))} | "
                f"{ms(c('proj', 'cosine'))} | {ms(c('proj', 'mae'))} |"
            )

    ig_path = "results/interface_generalization/results.json"
    if os.path.exists(ig_path):
        ig = json.load(open(ig_path, encoding="utf-8"))
        out += [
            "",
            "## Language-geometry interface generalization (`scripts/64_interface_generalization_study.py`, 3 seeds)",
            "",
            "All four interfaces share the same frozen 60-epoch QM9 `Encoder` (`checkpoints/v0/ylm_matscibert_s0.pt`) and frozen `MiniLM-L6-v2` token embeddings on molecule-disjoint official `TensorQA-v0.1.0` GFN2-xTB splits:",
            "",
            "| Interface | `rigid_single` cos | `ood_synonym_single` cos | `ood_synonym_comp` cos | `group_ood` cos | Rule parse rate (ID / OOD) |",
            "|---|---|---|---|---|---|",
        ]
        labels_ig = [
            ("Rule parser + O(3) executor (`rule_exec`)", "rule_exec"),
            ("Option A continuous (`optA_cont`)", "optA_cont"),
            ("Option B/C-lite continuous (`optBC_cont`)", "optBC_cont"),
            ("Neuro-symbolic + O(3) executor (`GroupProgModel`)", "group_prog"),
        ]
        for title, key in labels_ig:
            d = ig["interfaces"][key]
            pr = (
                f"{d['rigid_single']['parse_rate']:.3f} / {d['ood_synonym_single']['parse_rate']:.3f}"
                if "parse_rate" in d["rigid_single"]
                else "learned"
            )
            out.append(
                f"| {title} | {d['rigid_single']['cosine_mean']:.4f} ± {d['rigid_single']['cosine_std']:.4f} | "
                f"{d['ood_synonym_single']['cosine_mean']:.4f} ± {d['ood_synonym_single']['cosine_std']:.4f} | "
                f"{d['ood_synonym_comp']['cosine_mean']:.4f} ± {d['ood_synonym_comp']['cosine_std']:.4f} | "
                f"{d['group_ood']['cosine_mean']:.4f} ± {d['group_ood']['cosine_std']:.4f} | {pr} |"
            )

    sc_path = "results/scaling/results.json"
    if os.path.exists(sc_path):
        sc = json.load(open(sc_path, encoding="utf-8"))
        lat = {row["n_atoms"]: row["median_ms"] for row in sc["latency_ms"]}
        out += [
            "",
            "## Cross-attention parameter & system-size scaling (`scripts/66_scaling_benchmark.py`)",
            "",
            "| Architecture | Cross-attn params | Checkpoint (fp32) | `N=15` ms | `N=64` ms | `N=180` ms | `N=400` ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for mname in ("Option A", "Option C (channelwise)", "Option C (fully_connected)"):
            info = sc["models"][mname]
            out.append(
                f"| {mname} | {info['cross_attn_params']:,} | {info['checkpoint_mb_fp32']:.2f} MB | "
                f"{lat[15][mname]:.2f} | {lat[64][mname]:.2f} | {lat[180][mname]:.2f} | {lat[400][mname]:.2f} |"
            )

    open("results_baselines.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()

"""Compare 3D models side by side: YLM-Universe vs Uni-Mol, GraphMVP/3D-InfoMax, and EquiLLM.

Trains all four models on the same splits and optimizer settings, then tests them under
random 3D rotations and mirror flips (`O(3)`) on both `test` and `test_heldout`.

Examples:
  # Fast CPU smoke benchmark (~45s on CPU)
  python scripts/32_benchmark_vs_unimol.py --data_dir data/universe_smoke --tiny --epochs 20 --device cpu

  # Full GPU Foundation Model benchmark
  python scripts/32_benchmark_vs_unimol.py --data_dir data/universe_spice --ylm_backbone mace-mp-0-medium \
    --text_encoder matscibert --epochs 30 --batch_size 64 --eval_rotations 32 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fm_baselines import (  # noqa: E402
    EquiLLMMultiTaskBaseline,
    GraphMVP3DInfoMaxBaseline,
    UniMolStyle3DBaseline,
    match_baseline_d_model,
    match_equillm_d_model,
)
from src.universe_data import (  # noqa: E402
    QuestionFeaturizer,
    build_cpu_smoke_universe_dataset,
    load_universe_jsonl,
    make_universe_batches,
)
from src.universe_fm import (  # noqa: E402
    YLMUniverseModel,
    compute_universe_loss,
    coordinate_denoise_loss,
    evaluate_universe_o3,
)


def _load_splits(data_dir: str | None, seed: int) -> dict:
    if data_dir and Path(data_dir, "train.jsonl").exists():
        base = Path(data_dir)
        return {
            name: load_universe_jsonl(base / f"{name}.jsonl")
            for name in ("train", "val", "test", "test_heldout")
            if (base / f"{name}.jsonl").exists()
        }
    return build_cpu_smoke_universe_dataset(n_conformers_per_archetype=12, seed=seed)


def _train_one_model(
    label: str,
    model: torch.nn.Module,
    splits: dict,
    featurizer: QuestionFeaturizer,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    attn_loss_weight: float,
    denoise_weight: float,
    infonce_weight: float,
    eval_rotations: int,
    seed: int,
    device: str,
) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.1)

    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        batches = make_universe_batches(splits["train"], featurizer, batch_size=batch_size, shuffle=True, rng=rng, device=device)
        for batch in batches:
            opt.zero_grad()
            out = model(
                batch["Z"], batch["pos"], batch["atom_mask"],
                batch["text"], batch["text_mask"],
                lot_id=batch["lot_id"], cell=batch["cell"], pbc=batch["pbc"],
            )
            loss_dict = compute_universe_loss(
                out,
                task_kind=batch["task_kind"],
                scalar_target=batch["scalar_target"],
                vector_target=batch["vector_target"],
                tensor_target=batch["tensor_target"],
                target_atom_mask=batch["target_atom_mask"],
                lot_id=batch["lot_id"],
                attn_loss_weight=attn_loss_weight,
            )
            loss = loss_dict["loss"]
            if isinstance(model, GraphMVP3DInfoMaxBaseline) and infonce_weight > 0.0:
                c_loss = model.contrastive_3d_2d_loss(
                    batch["Z"], batch["pos"], batch["atom_mask"],
                    lot_id=batch["lot_id"], cell=batch["cell"], pbc=batch["pbc"],
                )
                loss = loss + infonce_weight * c_loss
            elif denoise_weight > 0.0:
                d_loss = coordinate_denoise_loss(
                    model, batch["Z"], batch["pos"], batch["atom_mask"],
                    lot_id=batch["lot_id"], cell=batch["cell"], pbc=batch["pbc"],
                )
                loss = loss + denoise_weight * d_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 2.0)
            opt.step()
        sched.step()

    elapsed = time.perf_counter() - t0
    test_b = make_universe_batches(splits["test"], featurizer, batch_size=batch_size, device=device)
    held_b = make_universe_batches(splits["test_heldout"], featurizer, batch_size=batch_size, device=device)
    m_test = evaluate_universe_o3(model, test_b, n_rotations=eval_rotations, include_reflections=True, device=device)
    m_held = evaluate_universe_o3(model, held_b, n_rotations=eval_rotations, include_reflections=True, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[{label:30s}] sc_mae={m_test['scalar_mae']:.4f} | "
        f"vec_cos={m_test['vector_cos']:+.4f} (eq={m_test['vector_eq_err']:.2e}) | "
        f"ten_cos={m_test['tensor_cos']:+.4f} (eq={m_test['tensor_eq_err']:.2e}) | "
        f"route={m_test['routing_acc']:.3f} | held_vec_cos={m_held['vector_cos']:+.4f} ({elapsed:.1f}s)"
    )
    return {
        "model": label,
        "n_params": n_params,
        "elapsed_sec": round(elapsed, 2),
        "test": m_test,
        "test_heldout": m_held,
    }


def _parse_seeds(seeds_arg: list[str] | None, default_seed: int) -> list[int]:
    if not seeds_arg:
        return [int(default_seed)]
    out: list[int] = []
    for item in seeds_arg:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out or [int(default_seed)]


def _aggregate_seed_runs(label: str, seed_runs: list[dict], seeds: list[int]) -> dict:
    def _mean_std_dict(split_key: str) -> tuple[dict[str, float], dict[str, float]]:
        keys = seed_runs[0][split_key].keys()
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for k in keys:
            vals = [float(r[split_key][k]) for r in seed_runs if isinstance(r[split_key].get(k), (int, float))]
            if vals:
                means[k] = float(np.mean(vals))
                stds[k] = float(np.std(vals))
        return means, stds

    test_mean, test_std = _mean_std_dict("test")
    held_mean, held_std = _mean_std_dict("test_heldout")
    return {
        "model": label,
        "n_params": seed_runs[0]["n_params"],
        "seeds": list(seeds),
        "elapsed_sec": round(float(sum(r["elapsed_sec"] for r in seed_runs)), 2),
        "test": test_mean,
        "test_std": test_std,
        "test_heldout": held_mean,
        "test_heldout_std": held_std,
        "per_seed": seed_runs,
    }


def _fmt_pm(mean: float, std: float, fmt: str, multi_seed: bool) -> str:
    if not multi_seed:
        return format(mean, fmt)
    std_fmt = fmt.lstrip("+")
    return f"{format(mean, fmt)} ± {format(std, std_fmt)}"


def _render_markdown(rows: list[dict], eval_rotations: int, seeds: list[int] | None = None, match_params: bool = False) -> str:
    seeds = seeds or rows[0].get("seeds", [42])
    multi = len(seeds) > 1
    seed_note = f"Seeds: `{', '.join(str(s) for s in seeds)}`" + (" (`mean ± std`)" if multi else " (single seed)")
    size_note = "size-matched (`--match_params`)" if match_params else "default widths"
    lines = [
        f"# 3D Model Comparison ($K={eval_rotations}$ Random $O(3)$ Rotations + Mirror Flips)",
        "",
        f"{seed_note}; baseline widths: {size_note}.",
        "",
        "## Table A: Invariant Scalar (`0e`) & Language-to-Atom Routing Performance",
        "",
        "| Model Family | Params | Scalar MAE (`0e`) ↓ | Scalar Rot-Inv Err ↓ | Routing Acc (Test) ↑ | Routing Acc (Held-out) ↑ |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        t = r["test"]
        ts = r.get("test_std", {})
        h = r["test_heldout"]
        hs = r.get("test_heldout_std", {})
        lines.append(
            f"| **{r['model']}** | {r['n_params']:,} | "
            f"{_fmt_pm(t['scalar_mae'], ts.get('scalar_mae', 0.0), '.4f', multi)} | "
            f"{_fmt_pm(t['scalar_inv_err'], ts.get('scalar_inv_err', 0.0), '.2e', multi)} | "
            f"{_fmt_pm(t['routing_acc'], ts.get('routing_acc', 0.0), '.3f', multi)} | "
            f"{_fmt_pm(h['routing_acc'], hs.get('routing_acc', 0.0), '.3f', multi)} |"
        )
    lines.extend([
        "",
        "## Table B: Covariant Vector (`1o`: T1/T3/T4/T5) & Symmetric Tensor (`0e+2e`: T2) Performance",
        "",
        "| Model Family | Params | Vec Cos (`1o`) ↑ | Force Cos (T3) ↑ | Comp Diff/Proj Cos ↑ | Vec $O(3)$ Eq Err ↓ | Tensor Cos (`0e+2e`) ↑ | Held-out Vec Cos ↑ |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for r in rows:
        t = r["test"]
        ts = r.get("test_std", {})
        h = r["test_heldout"]
        hs = r.get("test_heldout_std", {})
        per_seed = r.get("per_seed", [r])
        comp_per_seed = []
        for sr in per_seed:
            st = sr["test"]
            vals = [st[k] for k in ("diff_cos", "proj_cos") if k in st]
            comp_per_seed.append(float(sum(vals) / len(vals)) if vals else 0.0)
        comp_mean = float(np.mean(comp_per_seed))
        comp_std = float(np.std(comp_per_seed))
        lines.append(
            f"| **{r['model']}** | {r['n_params']:,} | "
            f"{_fmt_pm(t['vector_cos'], ts.get('vector_cos', 0.0), '+.4f', multi)} | "
            f"{_fmt_pm(t.get('force_cos', 0.0), ts.get('force_cos', 0.0), '+.4f', multi)} | "
            f"{_fmt_pm(comp_mean, comp_std, '+.4f', multi)} | "
            f"{_fmt_pm(t['vector_eq_err'], ts.get('vector_eq_err', 0.0), '.2e', multi)} | "
            f"{_fmt_pm(t['tensor_cos'], ts.get('tensor_cos', 0.0), '+.4f', multi)} | "
            f"{_fmt_pm(h['vector_cos'], hs.get('vector_cos', 0.0), '+.4f', multi)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Benchmark YLM-Universe against Uni-Mol, GraphMVP, and EquiLLM.")
    parser.add_argument("--data_dir", type=str, default="data/universe_smoke")
    parser.add_argument("--out_dir", type=str, default="runs/benchmark_vs_unimol")
    parser.add_argument("--ylm_backbone", type=str, default="e3nn")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--match_params", action="store_true", help="Widen baselines to match YLM parameter count.")
    parser.add_argument("--baseline_d_model", type=int, default=None, help="Explicit d_model width for baselines.")
    parser.add_argument("--text_encoder", type=str, default="fast", choices=["fast", "minilm", "matscibert"])
    parser.add_argument("--text_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--attn_loss_weight", type=float, default=0.30)
    parser.add_argument("--denoise_weight", type=float, default=0.15)
    parser.add_argument("--infonce_weight", type=float, default=0.20)
    parser.add_argument("--eval_rotations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=str, default=None, help="Seeds for mean +/- std, e.g. --seeds 42 43 44.")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    seeds = _parse_seeds(args.seeds, args.seed)
    splits = _load_splits(args.data_dir, seeds[0])
    featurizer = QuestionFeaturizer(mode=args.text_encoder, dim=args.text_dim)

    d_model = 64 if args.tiny else 128
    n_layers = 2 if args.tiny else 3
    n_vec = 16 if args.tiny else 32
    hidden_irreps = (
        "24x0e + 4x0o + 16x1o + 4x1e + 8x2e"
        if args.tiny
        else "64x0e + 8x0o + 32x1o + 8x1e + 16x2e"
    )
    cross_irreps = (
        "16x0e + 16x1o + 4x1e + 8x2e"
        if args.tiny
        else "32x0e + 32x1o + 8x1e + 16x2e"
    )

    def _make_ylm() -> torch.nn.Module:
        if args.ylm_backbone.lower().startswith("mace"):
            return YLMUniverseModel.from_mace(
                text_dim=featurizer.dim, backbone=args.ylm_backbone,
                cross_irreps=cross_irreps, n_cross_layers=2, option="C", c_tp_mode="channelwise", d_text=d_model * 2,
            )
        return YLMUniverseModel(
            text_dim=featurizer.dim, backbone="e3nn",
            hidden_irreps=hidden_irreps, cross_irreps=cross_irreps,
            n_layers=n_layers, n_cross_layers=2, option="C", c_tp_mode="channelwise", d_text=d_model * 2,
        )

    if args.baseline_d_model is not None:
        d_gmvp = d_unimol = d_equillm = int(args.baseline_d_model)
    elif args.match_params:
        ylm_params = sum(p.numel() for p in _make_ylm().parameters())
        d_gmvp = match_baseline_d_model(
            lambda d: GraphMVP3DInfoMaxBaseline(text_dim=featurizer.dim, d_model=d, n_layers=n_layers),
            ylm_params,
        )
        d_unimol = match_baseline_d_model(
            lambda d: UniMolStyle3DBaseline(text_dim=featurizer.dim, d_model=d, n_layers=n_layers, n_heads=4),
            ylm_params,
        )
        d_equillm = match_equillm_d_model(
            ylm_params,
            text_dim=featurizer.dim,
            hidden_irreps=hidden_irreps,
            n_layers=n_layers,
            n_vec=n_vec,
        )
    else:
        d_gmvp = d_unimol = d_equillm = d_model

    candidates = [
        (
            "GraphMVP / 3D-InfoMax",
            lambda: GraphMVP3DInfoMaxBaseline(text_dim=featurizer.dim, d_model=d_gmvp, n_layers=n_layers),
        ),
        (
            "Uni-Mol (3D Dist-Transformer)",
            lambda: UniMolStyle3DBaseline(text_dim=featurizer.dim, d_model=d_unimol, n_layers=n_layers, n_heads=4),
        ),
        (
            "EquiLLM (Inv-Bridge + 1o)",
            lambda: EquiLLMMultiTaskBaseline(
                text_dim=featurizer.dim, hidden_irreps=hidden_irreps, n_layers=n_layers,
                d_model=d_equillm, n_vec=n_vec,
            ),
        ),
        ("YLM-Universe FM (Ours)", _make_ylm),
    ]

    rows = []
    for label, make_mdl in candidates:
        seed_runs = []
        for s in seeds:
            res = _train_one_model(
                label, make_mdl(), splits, featurizer,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                attn_loss_weight=args.attn_loss_weight, denoise_weight=args.denoise_weight,
                infonce_weight=args.infonce_weight, eval_rotations=args.eval_rotations,
                seed=s, device=args.device,
            )
            res["seed"] = s
            seed_runs.append(res)
        rows.append(_aggregate_seed_runs(label, seed_runs, seeds))

    out_p = Path(args.out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    md_text = _render_markdown(rows, args.eval_rotations, seeds=seeds, match_params=args.match_params)
    (out_p / "benchmark_table.md").write_text(md_text, encoding="utf-8")
    payload = {"args": {**vars(args), "resolved_seeds": seeds}, "models": rows}
    (out_p / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + md_text)
    return payload


if __name__ == "__main__":
    main()

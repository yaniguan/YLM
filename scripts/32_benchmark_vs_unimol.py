"""Head-to-head Foundation Model benchmark: YLM-Universe vs Uni-Mol, GraphMVP/3D-InfoMax, and EquiLLM.

Trains all four architectures under identical splits, optimizer settings, auxiliary pretraining
losses (InfoNCE 3D-2D for GraphMVP; 3D coordinate denoising for Uni-Mol, EquiLLM, and YLM),
and evaluates under K-rotation O(3) + reflection protocol on both `test` and `test_heldout`.

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


def _render_markdown(rows: list[dict], eval_rotations: int) -> str:
    lines = [
        f"# Head-to-Head 3D Foundation Model Benchmark ($K={eval_rotations}$ Random $O(3)$ Rotations + Reflections)",
        "",
        "## Table A: Invariant Scalar (`0e`) & Language-to-Atom Routing Performance",
        "",
        "| Model Family | Params | Scalar MAE (`0e`) ↓ | Scalar Rot-Inv Err ↓ | Routing Acc (Test) ↑ | Routing Acc (Held-out) ↑ |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        t = r["test"]
        h = r["test_heldout"]
        lines.append(
            f"| **{r['model']}** | {r['n_params']:,} | {t['scalar_mae']:.4f} | "
            f"{t['scalar_inv_err']:.2e} | {t['routing_acc']:.3f} | {h['routing_acc']:.3f} |"
        )
    lines.extend([
        "",
        "## Table B: Covariant Vector (`1o`: T1/T3/T4/T5) & Symmetric Tensor (`0e+2e`: T2) Performance",
        "",
        "| Model Family | Vec Cos (`1o`) ↑ | Force Cos (T3) ↑ | Comp Diff/Proj Cos ↑ | Vec $O(3)$ Eq Err ↓ | Tensor Cos (`0e+2e`) ↑ | Held-out Vec Cos ↑ |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for r in rows:
        t = r["test"]
        h = r["test_heldout"]
        comp_vals = [t[k] for k in ("diff_cos", "proj_cos") if k in t]
        comp_cos = float(sum(comp_vals) / len(comp_vals)) if comp_vals else 0.0
        lines.append(
            f"| **{r['model']}** | {t['vector_cos']:+.4f} | {t.get('force_cos', 0.0):+.4f} | "
            f"{comp_cos:+.4f} | {t['vector_eq_err']:.2e} | {t['tensor_cos']:+.4f} | {h['vector_cos']:+.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Benchmark YLM-Universe against Uni-Mol, GraphMVP, and EquiLLM.")
    parser.add_argument("--data_dir", type=str, default="data/universe_smoke")
    parser.add_argument("--out_dir", type=str, default="runs/benchmark_vs_unimol")
    parser.add_argument("--ylm_backbone", type=str, default="e3nn")
    parser.add_argument("--tiny", action="store_true")
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
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    splits = _load_splits(args.data_dir, args.seed)
    featurizer = QuestionFeaturizer(mode=args.text_encoder, dim=args.text_dim)

    d_model = 64 if args.tiny else 128
    n_layers = 2 if args.tiny else 3
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

    candidates = [
        (
            "GraphMVP / 3D-InfoMax",
            GraphMVP3DInfoMaxBaseline(text_dim=featurizer.dim, d_model=d_model, n_layers=n_layers),
        ),
        (
            "Uni-Mol (3D Dist-Transformer)",
            UniMolStyle3DBaseline(text_dim=featurizer.dim, d_model=d_model, n_layers=n_layers, n_heads=4),
        ),
        (
            "EquiLLM (Inv-Bridge + 1o)",
            EquiLLMMultiTaskBaseline(
                text_dim=featurizer.dim, hidden_irreps=hidden_irreps, n_layers=n_layers,
                d_model=d_model, n_vec=(16 if args.tiny else 32),
            ),
        ),
        (
            "YLM-Universe FM (Ours)",
            (
                YLMUniverseModel.from_mace(
                    text_dim=featurizer.dim, backbone=args.ylm_backbone,
                    cross_irreps=cross_irreps, n_cross_layers=2, option="C", c_tp_mode="channelwise", d_text=d_model * 2,
                )
                if args.ylm_backbone.lower().startswith("mace")
                else YLMUniverseModel(
                    text_dim=featurizer.dim, backbone="e3nn",
                    hidden_irreps=hidden_irreps, cross_irreps=cross_irreps,
                    n_layers=n_layers, n_cross_layers=2, option="C", c_tp_mode="channelwise", d_text=d_model * 2,
                )
            ),
        ),
    ]

    rows = []
    for label, mdl in candidates:
        res = _train_one_model(
            label, mdl, splits, featurizer,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            attn_loss_weight=args.attn_loss_weight, denoise_weight=args.denoise_weight,
            infonce_weight=args.infonce_weight, eval_rotations=args.eval_rotations,
            seed=args.seed, device=args.device,
        )
        rows.append(res)

    out_p = Path(args.out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    md_text = _render_markdown(rows, args.eval_rotations)
    (out_p / "benchmark_table.md").write_text(md_text, encoding="utf-8")
    payload = {"args": vars(args), "models": rows}
    (out_p / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + md_text)
    return payload


if __name__ == "__main__":
    main()

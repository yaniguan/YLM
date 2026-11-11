"""Train and O(3)-evaluate YLMUniverseModel on multi-task (0e + 1o + 2e) Universe splits.

Examples:
  # Fast CPU smoke test (runs end-to-end in ~20s on CPU)
  python scripts/30_train_universe_fm.py --name universe_cpu_smoke --data_dir data/universe_smoke --tiny --epochs 20 --device cpu

  # GPU Foundation Model run mounted on pretrained MACE-MP-0 / MACE-OFF23
  python scripts/30_train_universe_fm.py --name ylm_mace_mp0 --data_dir data/universe_spice \
    --backbone mace-mp-0-medium --freeze_mace --unfreeze_mace_epoch 10 \
    --text_encoder matscibert --epochs 30 --batch_size 64 --device cuda
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

from src.mace_backbone import MACEEncoder  # noqa: E402
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


def _load_or_build_splits(data_dir: str | None, seed: int) -> dict:
    if data_dir and Path(data_dir, "train.jsonl").exists():
        base = Path(data_dir)
        out = {}
        for name in ("train", "val", "test", "test_heldout"):
            fp = base / f"{name}.jsonl"
            out[name] = load_universe_jsonl(fp) if fp.exists() else []
        return out
    return build_cpu_smoke_universe_dataset(n_conformers_per_archetype=12, seed=seed)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Train YLMUniverseModel multi-rank foundation model.")
    parser.add_argument("--name", type=str, default="universe_smoke")
    parser.add_argument("--data_dir", type=str, default="data/universe_smoke")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="e3nn", help="'e3nn' or 'mace-mp-0-{small,medium,large}' / 'mace-off23-*'")
    parser.add_argument("--freeze_mace", action="store_true", default=True)
    parser.add_argument("--unfreeze_mace_epoch", type=int, default=-1, help="Epoch to unfreeze MACE at 0.1x lr (-1 keeps frozen).")
    parser.add_argument("--option", type=str, default="C", choices=["A", "B", "C"])
    parser.add_argument("--c_tp_mode", type=str, default="channelwise", choices=["channelwise", "fully_connected"])
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_cross_layers", type=int, default=2)
    parser.add_argument("--tiny", action="store_true", help="Compact CPU-friendly irreps configuration.")
    parser.add_argument("--text_encoder", type=str, default="fast", choices=["fast", "minilm", "matscibert"])
    parser.add_argument("--text_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--attn_loss_weight", type=float, default=0.30)
    parser.add_argument("--denoise_weight", type=float, default=0.15)
    parser.add_argument("--eval_rotations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_ckpt", action="store_true", help="Save model state_dict to runs/<name>/checkpoint.pt")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    splits = _load_or_build_splits(args.data_dir, args.seed)
    featurizer = QuestionFeaturizer(mode=args.text_encoder, dim=args.text_dim)

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
    n_layers = 2 if args.tiny else args.n_layers
    d_text = 128 if args.tiny else 256

    if args.backbone.lower().startswith("mace"):
        model = YLMUniverseModel.from_mace(
            text_dim=featurizer.dim,
            backbone=args.backbone,
            freeze_mace=args.freeze_mace,
            cross_irreps=cross_irreps,
            n_cross_layers=args.n_cross_layers,
            option=args.option,
            c_tp_mode=args.c_tp_mode,
            d_text=d_text,
        ).to(args.device)
    else:
        model = YLMUniverseModel(
            text_dim=featurizer.dim,
            backbone="e3nn",
            hidden_irreps=hidden_irreps,
            cross_irreps=cross_irreps,
            n_layers=n_layers,
            n_cross_layers=args.n_cross_layers,
            option=args.option,
            c_tp_mode=args.c_tp_mode,
            d_text=d_text,
        ).to(args.device)

    val_batches = make_universe_batches(splits["val"], featurizer, batch_size=args.batch_size, device=args.device)
    test_batches = make_universe_batches(splits["test"], featurizer, batch_size=args.batch_size, device=args.device)
    held_batches = make_universe_batches(splits["test_heldout"], featurizer, batch_size=args.batch_size, device=args.device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.1)

    history = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if (
            args.unfreeze_mace_epoch > 0
            and epoch == args.unfreeze_mace_epoch
            and isinstance(model.encoder, MACEEncoder)
        ):
            model.encoder.freeze_mace = False
            model.encoder.mace.requires_grad_(True)
            optimizer.add_param_group({"params": list(model.encoder.mace.parameters()), "lr": args.lr * 0.05})

        model.train()
        train_batches = make_universe_batches(
            splits["train"], featurizer, batch_size=args.batch_size, shuffle=True, rng=rng, device=args.device
        )
        ep_losses = []
        ep_route = []
        for batch in train_batches:
            optimizer.zero_grad()
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
                attn_loss_weight=args.attn_loss_weight,
            )
            total_loss = loss_dict["loss"]
            if args.denoise_weight > 0.0:
                d_loss = coordinate_denoise_loss(
                    model, batch["Z"], batch["pos"], batch["atom_mask"],
                    lot_id=batch["lot_id"], cell=batch["cell"], pbc=batch["pbc"],
                )
                total_loss = total_loss + args.denoise_weight * d_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            ep_losses.append(float(total_loss.detach().cpu()))
            ep_route.append(float(loss_dict["routing_loss"].detach().cpu()))

        scheduler.step()
        if epoch == 1 or epoch == args.epochs or (epoch % max(1, args.epochs // 4) == 0):
            val_m = evaluate_universe_o3(model, val_batches, n_rotations=2, device=args.device)
            rec = {
                "epoch": epoch,
                "train_loss": float(np.mean(ep_losses)),
                "train_routing_loss": float(np.mean(ep_route)),
                "val_vector_cos": val_m["vector_cos"],
                "val_tensor_cos": val_m["tensor_cos"],
                "val_scalar_mae": val_m["scalar_mae"],
                "val_routing_acc": val_m["routing_acc"],
            }
            history.append(rec)
            print(
                f"[epoch {epoch:02d}/{args.epochs:02d}] loss={rec['train_loss']:.4f} "
                f"vec_cos={rec['val_vector_cos']:.4f} ten_cos={rec['val_tensor_cos']:.4f} "
                f"sc_mae={rec['val_scalar_mae']:.4f} route_acc={rec['val_routing_acc']:.3f}"
            )

    elapsed = time.perf_counter() - t0
    test_metrics = evaluate_universe_o3(
        model, test_batches, n_rotations=args.eval_rotations, include_reflections=True, device=args.device
    )
    heldout_metrics = evaluate_universe_o3(
        model, held_batches, n_rotations=args.eval_rotations, include_reflections=True, device=args.device
    )

    run_dir = Path(args.out_dir) if args.out_dir else Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    results = {
        "name": args.name,
        "backbone": args.backbone,
        "option": args.option,
        "c_tp_mode": args.c_tp_mode,
        "n_params": n_params,
        "elapsed_sec": round(elapsed, 2),
        "args": vars(args),
        "history": history,
        "test": test_metrics,
        "test_heldout": heldout_metrics,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.save_ckpt:
        torch.save({"state_dict": model.state_dict(), "args": vars(args)}, run_dir / "checkpoint.pt")
    print(f"Saved results to {run_dir / 'results.json'} ({elapsed:.1f}s, {n_params:,} params)")
    return results


if __name__ == "__main__":
    main()

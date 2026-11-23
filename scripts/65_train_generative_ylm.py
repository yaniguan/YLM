"""Train and evaluate a compact generative YLMCausalLM (<vec>) checkpoint on official TensorQA-v0.

Mounts the 60-epoch QM9 xTB equivariant Encoder from checkpoints/v0/ylm_minilm_s0.pt into
YLMCausalLM (mode="xattnA", vec_readout=True, CharTokenizer) and trains the causal decoder +
IrrepsAdapter + <vec> readout head to autoregressively emit text answers containing <vec>
together with exact O(3)-equivariant 3D dipole (Debye) and atom-force (eV/A) vectors.

Outputs:
  checkpoints/generative_v0/{model.pt, config.json, results.json, README.md}
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src  # noqa: E402, F401
from e3nn import o3  # noqa: E402
from src.tasks.common import (  # noqa: E402
    Example,
    Seg,
    collate,
    decode_with_vectors,
    model_inputs,
    save_run_model,
)
from src.tasks.tokenizer import CharTokenizer, VEC_TOKEN  # noqa: E402
from src.toolkit.ingest import AU_DIPOLE_TO_DEBYE, AU_FORCE_TO_EV_ANG  # noqa: E402
from src.ylm_lm import YLMCausalLM, YLMConfig  # noqa: E402


def make_examples(labels, questions, split: str, limit: int, seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    dip_qs = [q for q in questions if q["split"] == split and q["task"] == "dipole"]
    frc_qs = [q for q in questions if q["split"] == split and q["task"] == "force"]
    rng.shuffle(dip_qs)
    rng.shuffle(frc_qs)
    half = limit // 2
    selected = dip_qs[:half] + frc_qs[: limit - half]
    rng.shuffle(selected)

    examples = []
    for q in selected:
        lab = labels[int(q["mol"])]
        geom = {
            "Z": np.asarray(lab["numbers"], dtype=np.int64),
            "pos": np.asarray(lab["pos"], dtype=np.float64),
        }
        if q["task"] == "dipole":
            vec_phys = np.asarray(lab["dipole"], dtype=np.float64) * AU_DIPOLE_TO_DEBYE
            if not np.isfinite(vec_phys).all() or np.linalg.norm(vec_phys) < 0.05:
                continue
            ans = f" -> dipole {VEC_TOKEN} Debye"
        else:
            atom = int(q["atom"])
            vec_phys = np.asarray(lab["forces"][atom], dtype=np.float64) * AU_FORCE_TO_EV_ANG
            if not np.isfinite(vec_phys).all() or np.linalg.norm(vec_phys) < 0.05:
                continue
            sym = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}.get(int(lab["numbers"][atom]), "X")
            ans = f" -> force ({sym}) {VEC_TOKEN} eV/A"

        examples.append(
            Example(
                segs=[Seg(q["question"]), Seg(ans, "answer")],
                geom=geom,
                meta={"vec_targets": [vec_phys.tolist()], "task": q["task"], "question": q["question"]},
            )
        )
    return examples


@torch.no_grad()
def evaluate_generative(model: YLMCausalLM, tok: CharTokenizer, eval_examples: list[Example], device: str = "cpu") -> dict:
    model.eval()
    prompts = [Example([Seg(ex.meta["question"])], ex.geom) for ex in eval_examples]
    decoded = decode_with_vectors(model, tok, prompts, max_new=30, batch_size=32, device=device)

    vec_rate = 0
    by_task = {"dipole": {"cos": [], "mae": []}, "force": {"cos": [], "mae": []}}
    for ex, dec in zip(eval_examples, decoded):
        tgt = np.asarray(ex.meta["vec_targets"][0], dtype=np.float64)
        if dec["vectors"]:
            vec_rate += 1
            pred = np.asarray(dec["vectors"][0], dtype=np.float64)
        else:
            pred = np.zeros(3, dtype=np.float64)
        cos = float(np.dot(pred, tgt) / max(np.linalg.norm(pred) * np.linalg.norm(tgt), 1e-12))
        mae = float(np.abs(pred - tgt).mean())
        by_task[ex.meta["task"]]["cos"].append(cos)
        by_task[ex.meta["task"]]["mae"].append(mae)

    # Verify exact O(3) rotation + reflection equivariance of autoregressive <vec> output
    rot_errs, ref_errs = [], []
    sub = eval_examples[:16]
    p_base = [Example([Seg(ex.meta["question"])], ex.geom) for ex in sub]
    d_base = decode_with_vectors(model, tok, p_base, max_new=30, batch_size=16, device=device)
    for _ in range(2):
        R = o3.rand_matrix().numpy().astype(np.float64)
        for sign, bucket in ((1.0, rot_errs), (-1.0, ref_errs)):
            Q = sign * R
            p_rot = [
                Example([Seg(ex.meta["question"])], {"Z": ex.geom["Z"], "pos": ex.geom["pos"] @ Q.T})
                for ex in sub
            ]
            d_rot = decode_with_vectors(model, tok, p_rot, max_new=30, batch_size=16, device=device)
            for b0, br in zip(d_base, d_rot):
                if b0["vectors"] and br["vectors"]:
                    v0 = np.asarray(b0["vectors"][0], dtype=np.float64)
                    vr = np.asarray(br["vectors"][0], dtype=np.float64)
                    if np.linalg.norm(v0) > 1e-4:
                        bucket.append(float(np.linalg.norm(vr - v0 @ Q.T) / np.linalg.norm(v0)))

    return {
        "n_eval": len(eval_examples),
        "vec_emission_rate": round(vec_rate / max(len(eval_examples), 1), 4),
        "dipole": {
            "cosine": round(float(np.mean(by_task["dipole"]["cos"])), 4),
            "mae_debye": round(float(np.mean(by_task["dipole"]["mae"])), 4),
            "n": len(by_task["dipole"]["cos"]),
        },
        "force": {
            "cosine": round(float(np.mean(by_task["force"]["cos"])), 4),
            "mae_ev_ang": round(float(np.mean(by_task["force"]["mae"])), 4),
            "n": len(by_task["force"]["cos"]),
        },
        "o3_equivariance": {
            "mean_rotation_rel_err": float(np.mean(rot_errs)) if rot_errs else None,
            "mean_reflection_rel_err": float(np.mean(ref_errs)) if ref_errs else None,
        },
        "sample_generations": [
            {"question": eval_examples[i].meta["question"], "generated": decoded[i]["text"]}
            for i in range(min(4, len(eval_examples)))
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--v0-ckpt", default="checkpoints/v0/ylm_matscibert_s0.pt")
    ap.add_argument("--out-dir", default="checkpoints/generative_v0")
    ap.add_argument("--train-size", type=int, default=4800)
    ap.add_argument("--val-size", type=int, default=160)
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    labels = pickle.load(open(args.labels, "rb"))
    qs = json.load(open(args.questions, encoding="utf-8"))

    train_ex = make_examples(labels, qs, "train", args.train_size, seed=args.seed)
    val_ex = make_examples(labels, qs, "val", args.val_size, seed=args.seed + 1)
    test_ex = make_examples(labels, qs, "test", args.test_size, seed=args.seed + 2)
    print(f"Generative dataset: train={len(train_ex)} val={len(val_ex)} test={len(test_ex)}", flush=True)

    tok = CharTokenizer()
    cfg = YLMConfig(
        mode="xattnA",
        lm="tiny",
        vocab_size=tok.vocab_size,
        n_positions=256,
        n_layer=3,
        n_embd=256,
        n_head=4,
        bos_token_id=tok.bos_id,
        eos_token_id=tok.eos_id,
        encoder="small",
        n_elements=100,
        adapter_layers=[2],
        vec_readout=True,
        vec_token_id=tok.vec_id,
        vec_loss_weight=0.5,
        freeze_encoder=True,
    )
    model = YLMCausalLM(cfg)
    model.load_v0_encoder(args.v0_ckpt, freeze=True, load_cross_and_readout=True)
    with torch.no_grad():
        model.vec_head.weight.mul_(2.2)
    for adapter in model.adapters:
        adapter.gate.data.fill_(0.2)
    # Pre-encode frozen 3D atom features on padded N=29 tensors once for fast CPU training
    from src.ylm_lm import collate_geometry

    with torch.no_grad():
        all_geom = collate_geometry(
            [ex.geom for ex in train_ex] + [{"Z": np.ones(29, dtype=np.int64), "pos": np.zeros((29, 3))}]
        )
        x_chunks = []
        for s in range(0, len(train_ex), 128):
            sub_g = {k: v[s:s + 128] for k, v in all_geom.items()}
            x_chunks.append(model.encode_geometry(sub_g)[0])
        train_x = torch.cat(x_chunks, dim=0)
        train_amask = all_geom["atom_mask"][:len(train_ex)]

    trainable = [p for p in model.parameters() if p.requires_grad]
    print("Param counts:", model.param_counts(), "trainable:", sum(p.numel() for p in trainable), flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * math.ceil(len(train_ex) / args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=args.lr * 0.1)

    best_score = -1e9
    best_sd = None
    for ep in range(args.epochs):
        model.train()
        order = list(range(len(train_ex)))
        random.shuffle(order)
        losses = []
        for s in range(0, len(order), args.batch_size):
            batch_idx = order[s:s + args.batch_size]
            batch_ex = [train_ex[i] for i in batch_idx]
            b = collate(batch_ex, tok, max_len=192, loss_kinds={"answer"}, need_geom=False)
            b["labels"][b["input_ids"] == tok.eos_id] = tok.eos_id
            vt = torch.tensor([ex.meta["vec_targets"][0] for ex in batch_ex], dtype=torch.float32)
            idx_t = torch.tensor(batch_idx, dtype=torch.long)
            max_n = int(train_amask[idx_t].sum(dim=1).max().item())
            gf = (train_x[idx_t, :max_n], train_amask[idx_t, :max_n])
            mi = model_inputs(b)
            mi.pop("geom", None)
            out = model(**mi, geom_feats=gf, vec_targets=vt)
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 2.0)
            opt.step()
            sched.step()
            losses.append(float(loss.item()))

        val_m = evaluate_generative(model, tok, val_ex[:80])
        score = val_m["vec_emission_rate"] + val_m["dipole"]["cosine"] + val_m["force"]["cosine"]
        print(
            f"epoch {ep:02d} | loss={np.mean(losses):.4f} | val vec_rate={val_m['vec_emission_rate']:.3f} "
            f"dip_cos={val_m['dipole']['cosine']:.3f} frc_cos={val_m['force']['cosine']:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

    assert best_sd is not None
    model.load_state_dict(best_sd)
    test_m = evaluate_generative(model, tok, test_ex)
    held_ex = make_examples(labels, qs, "test_heldout", args.test_size, seed=args.seed + 3)
    held_m = evaluate_generative(model, tok, held_ex)

    out_dir = Path(args.out_dir)
    save_run_model(
        str(out_dir),
        model,
        {
            "model_cfg": cfg.to_dict(),
            "args": {"arm": "xattnA_vec", "seed": args.seed, "epochs": args.epochs, "train_size": len(train_ex)},
            "units": {"dipole": "Debye", "force": "eV/A"},
        },
    )
    results = {
        "status": "MEASURED",
        "checkpoint": str(out_dir).replace("\\", "/"),
        "param_counts": model.param_counts(),
        "adapter_gates": model.gates(),
        "test": test_m,
        "test_heldout": held_m,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("Saved generative <vec> checkpoint & evaluation:", json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

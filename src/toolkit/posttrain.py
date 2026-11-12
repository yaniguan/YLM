"""Three-tier covariant physics labeler and IrrepsAdapter + <vec> SFT / DPO post-training pipeline."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch

from src.tasks.common import Example, Seg, collate, model_inputs, save_run_model, seed_everything
from src.tasks.sft import PreferencePair, dpo_loss, sequence_logps
from src.tasks.tokenizer import CharTokenizer, VEC_TOKEN
from src.toolkit.ingest import (
    AU_DIPOLE_TO_DEBYE,
    AU_FORCE_TO_EV_ANG,
    BOHR_TO_ANGSTROM,
    StructureFrame,
    covalent_radius,
    load_trajectory,
    symbol_for_z,
)
from src.ylm_lm import YLMCausalLM, YLMConfig

# Pauling electronegativities for analytical O(3)-covariant dipole estimation.
_PAULING_CHI = {
    1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    11: 0.93, 12: 1.31, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    19: 0.82, 20: 1.00, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65,
    35: 2.96, 53: 2.66,
}


def analytical_forces_and_dipole(frame: StructureFrame) -> tuple[np.ndarray, np.ndarray]:
    """Zero-dependency O(3)-covariant pairwise force (eV/A) and bond-charge dipole (Debye) estimator."""
    n = frame.n_atoms
    pos = frame.pos_angstrom
    forces = np.zeros((n, 3), dtype=np.float64)
    charges = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return forces, np.zeros(3, dtype=np.float64)

    for i in range(n):
        zi = int(frame.Z[i])
        chi_i = _PAULING_CHI.get(zi, 2.0)
        for j in range(i + 1, n):
            zj = int(frame.Z[j])
            chi_j = _PAULING_CHI.get(zj, 2.0)
            rij_vec = pos[i] - pos[j]
            r = float(np.linalg.norm(rij_vec))
            if r < 1e-8:
                continue
            r_hat = rij_vec / r
            r0 = covalent_radius(zi) + covalent_radius(zj)
            if r < 1.45 * r0:
                # Harmonic bond spring: F_i = -k * (r - r0) * (r_i - r_j)/r
                f_mag = -8.5 * (r - r0)
            elif r < 3.6:
                # Soft short-range steric repulsion
                sigma = 0.85 * (r0 + 0.9)
                f_mag = 1.2 * ((sigma / r) ** 7)
            else:
                f_mag = 0.0
            f_vec = f_mag * r_hat
            forces[i] += f_vec
            forces[j] -= f_vec

            # Antisymmetric pairwise bond-charge transfer ensures sum(charges) == 0 (translation-invariant dipole!).
            dq = 0.35 * np.tanh(0.8 * (chi_i - chi_j)) * np.exp(-((r / max(1.25 * r0, 0.5)) ** 2))
            charges[i] -= dq
            charges[j] += dq

    centroid = pos.mean(axis=0)
    dipole_debye = (charges[:, None] * (pos - centroid)).sum(axis=0) * 4.8032047
    return forces, dipole_debye


class PhysicsLabeler:
    """Three-tier labeler: 'extxyz' (recorded) -> 'xtb' (optional tblite) -> 'mmff'/'analytical' (offline)."""

    def __init__(self, backend: str = "auto"):
        if backend not in ("auto", "analytical", "mmff", "xtb", "extxyz"):
            raise ValueError(f"Unsupported backend {backend!r}")
        self.backend = backend

    def label_frame(self, frame: StructureFrame) -> StructureFrame:
        out = deepcopy(frame)
        if self.backend in ("auto", "extxyz") and out.forces_ev_ang is not None:
            if out.dipole_debye is None:
                _, dip = analytical_forces_and_dipole(out)
                out.dipole_debye = dip
            return out

        if self.backend == "extxyz" and out.forces_ev_ang is None:
            raise ValueError("Frame has no recorded forces for backend='extxyz'")

        if self.backend in ("auto", "xtb"):
            try:
                import tblite.interface as tb
                calc = tb.Calculator("GFN2-xTB", out.Z, out.pos_bohr)
                calc.set("verbosity", 0)
                res = calc.singlepoint()
                grad = np.asarray(res.get("gradient"), dtype=np.float64)
                dip = np.asarray(res.get("dipole"), dtype=np.float64)
                out.forces_ev_ang = -grad * AU_FORCE_TO_EV_ANG
                out.dipole_debye = dip * AU_DIPOLE_TO_DEBYE
                return out
            except Exception:
                if self.backend == "xtb":
                    raise

        # Analytical pairwise model (strictly O(3)-equivariant and available on every platform).
        forces, dipole = analytical_forces_and_dipole(out)
        out.forces_ev_ang = forces
        out.dipole_debye = dipole
        return out


@dataclass
class DirectionalSFTRecord:
    example: Example
    vec_target: np.ndarray
    atom_index: Optional[int]
    kind: str


def _atom_descriptors(frame: StructureFrame) -> list[tuple[int, str]]:
    """Return O(3)-invariant natural-language descriptors for uniquely identifiable atoms in a frame."""
    pos = frame.pos_angstrom
    n = frame.n_atoms
    descs = []
    raw_keys = []
    for i in range(n):
        zi = int(frame.Z[i])
        nb = []
        for j in range(n):
            if i == j:
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d <= 1.35 * (covalent_radius(zi) + covalent_radius(int(frame.Z[j]))):
                nb.append(symbol_for_z(int(frame.Z[j])))
        nb_str = ",".join(f"{cnt}{sym}" for sym, cnt in sorted(Counter(nb).items())) or "isolated"
        key = f"{symbol_for_z(zi)}({nb_str})"
        raw_keys.append(key)
    counts = Counter(raw_keys)
    for i, key in enumerate(raw_keys):
        if counts[key] == 1:
            descs.append((i, key))
    # Always include at least the highest-force atom if symmetry makes all topological keys degenerate.
    if not descs and n > 0 and frame.forces_ev_ang is not None:
        f_norms = np.linalg.norm(frame.forces_ev_ang, axis=-1)
        top = int(np.argmax(f_norms))
        descs.append((top, f"peak-force {symbol_for_z(int(frame.Z[top]))}"))
    return descs


def build_directional_sft_examples(
    frames: Sequence[StructureFrame],
    backend: str = "auto",
    max_atoms_per_frame: int = 3,
) -> list[DirectionalSFTRecord]:
    """Generate O(3)-invariant text prompts + <vec> answers paired with 3D covariant targets."""
    labeler = PhysicsLabeler(backend=backend)
    records: list[DirectionalSFTRecord] = []
    for raw_f in frames:
        f = labeler.label_frame(raw_f)
        geom = f.to_geom_dict()
        if f.dipole_debye is not None:
            mag = float(np.linalg.norm(f.dipole_debye))
            ex = Example(
                [Seg("Q: Give the dipole magnitude and direction of molecule? A: "),
                 Seg(f"|mu|={mag:.2f}D dir={VEC_TOKEN}", "answer")],
                geom,
                {"kind": "dipole", "dipole_mag": mag},
            )
            records.append(DirectionalSFTRecord(ex, f.dipole_debye.copy(), None, "dipole"))

        if f.forces_ev_ang is not None:
            for atom_idx, desc in _atom_descriptors(f)[:max_atoms_per_frame]:
                fvec = f.forces_ev_ang[atom_idx]
                fmag = float(np.linalg.norm(fvec))
                ex = Example(
                    [Seg(f"Q: Force on {desc}? A: "),
                     Seg(f"|F|={fmag:.2f}eV/A vec={VEC_TOKEN}", "answer")],
                    geom,
                    {"kind": "force", "atom_index": atom_idx, "force_mag": fmag},
                )
                records.append(DirectionalSFTRecord(ex, fvec.copy(), atom_idx, "force"))
    return records


def build_directional_dpo_pairs(frames: Sequence[StructureFrame], backend: str = "auto") -> list[PreferencePair]:
    """Construct O(3)-invariant counterfactual directional preference pairs (toward vs away from nearest neighbor)."""
    labeler = PhysicsLabeler(backend=backend)
    pairs: list[PreferencePair] = []
    for raw_f in frames:
        f = labeler.label_frame(raw_f)
        if f.n_atoms < 2 or f.forces_ev_ang is None:
            continue
        geom = f.to_geom_dict()
        pos = f.pos_angstrom
        for atom_idx, desc in _atom_descriptors(f)[:2]:
            dists = [(float(np.linalg.norm(pos[j] - pos[atom_idx])), j) for j in range(f.n_atoms) if j != atom_idx]
            _, nb_idx = min(dists, key=lambda item: item[0])
            bond_dir = pos[nb_idx] - pos[atom_idx]
            proj = float(np.dot(f.forces_ev_ang[atom_idx], bond_dir))
            true_word = "toward" if proj >= 0.0 else "away from"
            false_word = "away from" if proj >= 0.0 else "toward"
            nb_sym = symbol_for_z(int(f.Z[nb_idx]))
            prompt = [Seg(f"Q: Does the net force on {desc} pull toward or away from nearest {nb_sym}? A: ")]
            chosen = Example(prompt + [Seg(f"pulls {true_word} {nb_sym}", "answer")], geom)
            rejected = Example(prompt + [Seg(f"pulls {false_word} {nb_sym}", "answer")], geom)
            pairs.append(PreferencePair(chosen=chosen, rejected=rejected))
    return pairs


def run_posttrain(
    input_source: Union[str, Path, Sequence[StructureFrame]],
    out_dir: Union[str, Path],
    *,
    backend: str = "auto",
    mode: str = "xattnC",
    c_tp_mode: str = "channelwise",
    lm: str = "tiny",
    steps: int = 25,
    dpo_steps: int = 0,
    batch_size: int = 4,
    lr: float = 2e-3,
    freeze_lm: bool = False,
    seed: int = 0,
    device: str = "cpu",
    n_layer: Optional[int] = None,
    n_embd: Optional[int] = None,
    n_head: Optional[int] = None,
) -> dict:
    """Train YLMCausalLM + IrrepsAdapter + <vec> readout on directional SFT (and optional DPO) and save run_dir."""
    seed_everything(seed)
    frames = load_trajectory(input_source)
    has_motion = any(f.delta_pos_bohr is not None for f in frames)
    records = build_directional_sft_examples(frames, backend=backend)
    if not records:
        raise ValueError("No directional training records generated from input")

    tok = CharTokenizer()
    cfg = YLMConfig(
        mode=mode,
        lm=lm,
        vocab_size=tok.vocab_size,
        bos_token_id=tok.bos_id,
        eos_token_id=tok.eos_id,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        c_tp_mode=c_tp_mode,
        vec_readout=True,
        vec_token_id=tok.vec_id,
        use_motion_1o=has_motion,
    )
    model = YLMCausalLM(cfg).to(device)
    if freeze_lm:
        model.lm.requires_grad_(False)
        # Keep output head trainable only if vocabulary needs <vec> calibration, or rely on adapters + vec_head.
        for adapter in model.adapters:
            adapter.gate.data.fill_(0.2)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    model.train()

    losses = []
    vec_losses = []
    rng = np.random.default_rng(seed)
    for _ in range(max(1, steps)):
        idxs = rng.integers(0, len(records), size=min(batch_size, len(records)))
        chunk = [records[int(i)] for i in idxs]
        batch = collate([r.example for r in chunk], tok, max_len=160, loss_kinds=["answer"], device=device)
        targets = torch.as_tensor(np.stack([r.vec_target for r in chunk], axis=0),
                                  dtype=torch.float32, device=device)
        opt.zero_grad()
        out = model(**model_inputs(batch), vec_targets=targets)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 2.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        vec_losses.append(float(out["vec_loss"].detach().cpu()))

    dpo_history = []
    if dpo_steps > 0:
        pairs = build_directional_dpo_pairs(frames, backend=backend)
        if pairs:
            reference = deepcopy(model).eval()
            reference.requires_grad_(False)
            dpo_opt = torch.optim.AdamW(trainable, lr=lr * 0.25)
            for _ in range(dpo_steps):
                p_idxs = rng.integers(0, len(pairs), size=min(batch_size, len(pairs)))
                chosen_b = collate([pairs[int(i)].chosen for i in p_idxs], tok, max_len=160,
                                   loss_kinds=["answer"], device=device)
                reject_b = collate([pairs[int(i)].rejected for i in p_idxs], tok, max_len=160,
                                   loss_kinds=["answer"], device=device)
                with torch.no_grad():
                    ref_c = sequence_logps(reference, chosen_b)
                    ref_r = sequence_logps(reference, reject_b)
                pol_c = sequence_logps(model, chosen_b)
                pol_r = sequence_logps(model, reject_b)
                dloss = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
                dpo_opt.zero_grad()
                dloss.backward()
                dpo_opt.step()
                dpo_history.append(float(dloss.detach().cpu()))

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "stage": "directional_posttrain",
        "backend": backend,
        "model_cfg": cfg.to_dict(),
        "args": {"arm": mode, "freeze_lm": freeze_lm, "steps": steps, "dpo_steps": dpo_steps},
    }
    save_run_model(str(out_p), model, config_payload)
    summary = {
        "n_sft_records": len(records),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "initial_vec_loss": vec_losses[0],
        "final_vec_loss": vec_losses[-1],
        "dpo_losses": dpo_history,
        "out_dir": str(out_p),
    }
    (out_p / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

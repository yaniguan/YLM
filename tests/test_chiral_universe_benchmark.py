"""Tests for `0o` chiral pseudoscalar equivariance (`YLMUniverseModel`) and enantiomer benchmark."""
from __future__ import annotations

import numpy as np
import torch

from src.fm_baselines import (
    EquiLLMMultiTaskBaseline,
    GraphMVP3DInfoMaxBaseline,
    UniMolStyle3DBaseline,
)
from src.toolkit.ouroboros_bridge import embed_stereo_smiles, ouroboros_frames_to_universe_records
from src.universe_data import QuestionFeaturizer, UniverseRecord, collate_universe_records, compute_chiral_centers
from src.universe_fm import YLMUniverseModel, compute_universe_loss, evaluate_universe_o3
from e3nn import o3  # noqa: E402


def test_universe_model_0o_pseudoscalar_equivariance_and_planar_zero():
    """`out['chiral']` must obey `f(X Q^T) == det(Q) * f(X)` (`< 1e-10` in float64) and vanish on planar C2v/Cs."""
    torch.manual_seed(7)
    model = YLMUniverseModel(
        text_dim=32,
        backbone="e3nn",
        hidden_irreps="16x0e + 4x0o + 8x1o + 4x1e + 4x2e",
        cross_irreps="8x0e + 8x1o + 4x1e + 4x2e",
        n_layers=3,
        n_cross_layers=2,
        option="C",
        c_tp_mode="channelwise",
        d_text=32,
    ).to(dtype=torch.float64).eval()

    # 1) Chiral (R)-bromochlorofluoromethane conformer
    fr_r = embed_stereo_smiles("F[C@H](Cl)Br")
    Z = torch.tensor([fr_r.Z.tolist()], dtype=torch.long)
    pos = torch.tensor([fr_r.pos_bohr.tolist()], dtype=torch.float64)
    atom_mask = torch.ones_like(Z, dtype=torch.bool)
    text = torch.randn(1, 6, 32, dtype=torch.float64)
    text_mask = torch.ones(1, 6, dtype=torch.bool)
    lot_id = torch.tensor([1], dtype=torch.long)

    with torch.no_grad():
        out0 = model(Z, pos, atom_mask, text, text_mask, lot_id=lot_id)
        c0 = out0["chiral"]
        assert c0.abs().item() > 1e-6, "Expected non-zero 0o chiral response on chiral tetrahedron"

        # Proper rotation (det = +1) -> identical chiral sign & magnitude
        R_proper = o3.rand_matrix().to(dtype=torch.float64)
        if torch.linalg.det(R_proper) < 0:
            R_proper = -R_proper
        out_prop = model(Z, pos @ R_proper.T, atom_mask, text, text_mask, lot_id=lot_id)
        assert (out_prop["chiral"] - c0).abs().max().item() < 1e-10

        # Improper reflection/inversion (det = -1) -> exact sign flip `-c0`
        Q_improper = -o3.rand_matrix().to(dtype=torch.float64)
        if torch.linalg.det(Q_improper) > 0:
            Q_improper = -Q_improper
        out_imp = model(Z, pos @ Q_improper.T, atom_mask, text, text_mask, lot_id=lot_id)
        assert (out_imp["chiral"] + c0).abs().max().item() < 1e-10

        # 2a) Exact planar C2v water rotated arbitrarily in 3D -> machine-precision zero (< 1e-12)
        Z_w = torch.tensor([[8, 1, 1]], dtype=torch.long)
        pos_w = torch.tensor([[[0.0, 0.0, 0.0], [1.81, 0.0, 0.0], [-0.45, 1.75, 0.0]]], dtype=torch.float64) @ R_proper.T
        mask_w = torch.ones_like(Z_w, dtype=torch.bool)
        out_water = model(Z_w, pos_w, mask_w, text, text_mask, lot_id=lot_id)
        assert out_water["chiral"].abs().max().item() < 1e-12

        # 2b) Achiral cis-1,2-difluoroethene from ETKDGv3/MMFF rotated in 3D -> near-zero (< 1e-9)
        fr_cis = embed_stereo_smiles(r"F/C=C\F")
        assert compute_chiral_centers(fr_cis) == []
        Z_p = torch.tensor([fr_cis.Z.tolist()], dtype=torch.long)
        pos_p = torch.tensor([fr_cis.pos_bohr.tolist()], dtype=torch.float64) @ R_proper.T
        mask_p = torch.ones_like(Z_p, dtype=torch.bool)
        out_planar = model(Z_p, pos_p, mask_p, text, text_mask, lot_id=lot_id)
        assert out_planar["chiral"].abs().max().item() < 1e-9


def test_enantiomer_separation_ylm_vs_invariant_baselines():
    """`Uni-Mol`, `GraphMVP`, and `EquiLLM` stay <= 50% on paired (R)/(S) enantiomers; `YLMUniverseModel` hits 100%."""
    torch.manual_seed(11)
    featurizer = QuestionFeaturizer(mode="fast", dim=64)
    fr_r = embed_stereo_smiles("O=C1OC[C@H](F)O1")
    fr_s = embed_stereo_smiles("O=C1OC[C@@H](F)O1")
    c_r = compute_chiral_centers(fr_r)[0]
    c_s = compute_chiral_centers(fr_s)[0]
    assert c_r["handedness_sign"] == -c_s["handedness_sign"]

    zero_vec = [0.0, 0.0, 0.0]
    zero_ten = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    q = "[LOT=wb97m] What is the (R)/(S) chiral handedness pseudoscalar of this stereoisomer?"

    rng = np.random.default_rng(11)
    records: list[UniverseRecord] = []
    for k in range(4):
        rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rot) < 0:
            rot[:, 0] = -rot[:, 0]
        for tag, fr, c_info in (("R", fr_r, c_r), ("S", fr_s, c_s)):
            pos_rot = (fr.pos_bohr @ rot.T).tolist()
            records.append(UniverseRecord(
                structure_id=f"fec_{tag}_{k}",
                domain="ouroboros_chiral",
                lot="wb97m",
                split="train",
                Z=fr.Z.tolist(),
                pos_bohr=pos_rot,
                cell_bohr=None,
                pbc=None,
                question=q,
                task_kind="chiral",
                subtask="chiral",
                scalar_target=float(c_info["handedness_sign"]),
                vector_target=zero_vec,
                tensor_target=zero_ten,
                target_atoms=[int(c_info["atom_index"])],
            ))

    batch = collate_universe_records(records, featurizer)

    # Verify Ouroboros -> Universe chiral record generation emits `"chiral"` rows automatically
    ob_recs = ouroboros_frames_to_universe_records([fr_r, fr_s], split_ratios=(1.0, 0.0, 0.0))
    assert any(r.task_kind == "chiral" for r in ob_recs)

    # Train each baseline & YLMUniverseModel for 25 steps on paired enantiomers and evaluate under O(3)
    for baseline in (
        UniMolStyle3DBaseline(text_dim=64, d_model=32, n_layers=2, n_heads=4),
        GraphMVP3DInfoMaxBaseline(text_dim=64, d_model=32, n_layers=2),
        EquiLLMMultiTaskBaseline(
            text_dim=64,
            hidden_irreps="16x0e + 4x0o + 8x1o + 4x1e + 4x2e",
            n_layers=2,
            d_model=32,
            n_vec=8,
        ),
    ):
        opt_b = torch.optim.AdamW(baseline.parameters(), lr=3e-3)
        for _ in range(20):
            opt_b.zero_grad()
            out_b = baseline(batch["Z"], batch["pos"], batch["atom_mask"], batch["text"], batch["text_mask"], lot_id=batch["lot_id"])
            loss_b = compute_universe_loss(
                out_b,
                task_kind=batch["task_kind"],
                scalar_target=batch["scalar_target"],
                vector_target=batch["vector_target"],
                tensor_target=batch["tensor_target"],
                target_atom_mask=batch["target_atom_mask"],
                lot_id=batch["lot_id"],
            )["loss"]
            loss_b.backward()
            opt_b.step()
        m_b = evaluate_universe_o3(baseline, [batch], n_rotations=4, include_reflections=True)
        assert m_b["chiral_sign_acc"] <= 0.5001, f"0e model cannot exceed 50% enantiomer ceiling, got {m_b}"

    ylm = YLMUniverseModel(
        text_dim=64,
        backbone="e3nn",
        hidden_irreps="16x0e + 4x0o + 8x1o + 4x1e + 4x2e",
        cross_irreps="8x0e + 8x1o + 4x1e + 4x2e",
        n_layers=2,
        n_cross_layers=2,
        option="C",
        c_tp_mode="channelwise",
        d_text=64,
    )
    opt_y = torch.optim.AdamW(ylm.parameters(), lr=3e-3)
    for _ in range(25):
        opt_y.zero_grad()
        out_y = ylm(batch["Z"], batch["pos"], batch["atom_mask"], batch["text"], batch["text_mask"], lot_id=batch["lot_id"])
        loss_y = compute_universe_loss(
            out_y,
            task_kind=batch["task_kind"],
            scalar_target=batch["scalar_target"],
            vector_target=batch["vector_target"],
            tensor_target=batch["tensor_target"],
            target_atom_mask=batch["target_atom_mask"],
            lot_id=batch["lot_id"],
        )["loss"]
        loss_y.backward()
        opt_y.step()

    m_y = evaluate_universe_o3(ylm, [batch], n_rotations=6, include_reflections=True)
    assert m_y["chiral_sign_acc"] >= 0.999
    assert m_y["chiral_parity_err"] < 1e-5

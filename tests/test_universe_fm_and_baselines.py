"""Unit tests for YLMUniverseModel, Universe data pipeline, and Foundation Model baselines."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from e3nn import o3

from src.fm_baselines import (
    EquiLLMMultiTaskBaseline,
    GraphMVP3DInfoMaxBaseline,
    UniMolStyle3DBaseline,
    info_nce_3d_2d_loss,
)
from src.tensor_readout import rotate_matrix
from src.toolkit.ingest import ANGSTROM_TO_BOHR, StructureFrame
from src.universe_data import (
    QuestionFeaturizer,
    build_cpu_smoke_universe_dataset,
    collate_universe_records,
    compute_covariant_universe_labels,
    generate_diverse_queries,
)
from src.universe_fm import (
    LOT_TO_ID,
    YLMUniverseModel,
    compute_universe_loss,
    coordinate_denoise_loss,
    decoupled_vector_loss,
    evaluate_universe_o3,
    routing_supervision_loss,
)


class _DummyMACE(nn.Module):
    """Lightweight O(3)-equivariant mock MACE module with only 0e node_feats + conservative forces."""

    def __init__(self, atomic_numbers=(1, 3, 6, 8, 15, 26), r_max=5.0, feat_dim=8):
        super().__init__()
        self.node_feats_irreps = f"{feat_dim}x0e"
        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.long))
        self.register_buffer("r_max", torch.tensor(float(r_max)))
        self.node_proj = nn.Linear(len(atomic_numbers), feat_dim, bias=False)

    def forward(self, data: dict[str, torch.Tensor], training: bool = False, compute_force: bool = True):
        pos = data["positions"]
        attrs = data["node_attrs"]
        src, dst = data["edge_index"][0], data["edge_index"][1]
        shifts = data["shifts"]
        rel = pos[dst] - (pos[src] + shifts)
        dist2 = (rel * rel).sum(dim=-1, keepdim=True)
        edge_e = 25.0 * torch.exp(-0.3 * dist2).squeeze(-1)

        n_graphs = int(data["ptr"].numel() - 1)
        g_energy = torch.zeros(n_graphs, device=pos.device, dtype=pos.dtype)
        g_energy.index_add_(0, data["batch"][dst], edge_e)

        agg = torch.zeros(pos.shape[0], 1, device=pos.device, dtype=pos.dtype)
        agg.index_add_(0, dst, edge_e[:, None])
        node_feats = self.node_proj(attrs) * (1.0 + 0.2 * agg)

        forces = None
        if compute_force and pos.requires_grad:
            grad = torch.autograd.grad(
                [g_energy.sum()], [pos], create_graph=training, retain_graph=training, allow_unused=True
            )[0]
            forces = -grad if grad is not None else torch.zeros_like(pos)
        return {"energy": g_energy, "node_feats": node_feats, "forces": forces}


def _load_script(rel_path: str, mod_name: str):
    full = Path(__file__).resolve().parents[1] / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, full)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("reflection", [False, True])
def test_universe_model_exact_o3_equivariance_mixed_pbc_and_heavy_elements(reflection: bool):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(7)
        model = YLMUniverseModel(
            text_dim=16,
            backbone="e3nn",
            hidden_irreps="8x0e + 2x0o + 6x1o + 2x1e + 4x2e",
            cross_irreps="8x0e + 6x1o + 2x1e + 4x2e",
            n_layers=2,
            n_cross_layers=2,
            option="C",
            c_tp_mode="channelwise",
            d_text=32,
        ).eval()
        # Calibrate LOT scales to distinct values so LOT conditioning is actively tested under O(3)
        with torch.no_grad():
            model.lot_vec_logscale.weight.normal_(0.0, 0.4)
            model.lot_ten_logscale.weight.normal_(0.0, 0.4)
            model.lot_scalar_scale.weight.normal_(0.0, 0.3)
            model.lot_scalar_shift.weight.normal_(0.0, 0.3)

        # Row 0: heavy-element molecule (P, S, Cl, Br, I); Row 1: 3D PBC crystal (Li, Fe, P, O)
        Z = torch.tensor([[15, 16, 17, 35, 53], [3, 26, 15, 8, 0]], dtype=torch.long)
        pos = torch.randn(2, 5, 3, dtype=torch.float64) * 2.2
        atom_mask = torch.tensor([[True, True, True, True, True], [True, True, True, True, False]])
        cell = torch.zeros(2, 3, 3, dtype=torch.float64)
        cell[1] = torch.tensor([[8.5, 0.2, 0.0], [0.0, 8.7, 0.1], [0.1, 0.0, 8.9]], dtype=torch.float64)
        pbc = torch.tensor([[False, False, False], [True, True, True]])

        text = torch.randn(2, 4, 16, dtype=torch.float64)
        text_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
        lot_id = torch.tensor([LOT_TO_ID["wb97m"], LOT_TO_ID["pbe_u"]], dtype=torch.long)

        with torch.random.fork_rng():
            torch.manual_seed(19)
            R = o3.rand_matrix().to(torch.float64)
        if reflection:
            R = -R
        shift = torch.tensor([1.7, -2.3, 0.9], dtype=torch.float64)

        out0 = model(Z, pos, atom_mask, text, text_mask, lot_id=lot_id, cell=cell, pbc=pbc)
        out_r = model(Z, pos @ R.T + shift, atom_mask, text, text_mask, lot_id=lot_id, cell=cell @ R.T, pbc=pbc)

        torch.testing.assert_close(out_r["scalar"], out0["scalar"], atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(out_r["attn"], out0["attn"], atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(out_r["vector"], out0["vector"] @ R.T, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(out_r["tensor"], rotate_matrix(out0["tensor"], R), atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(out_r["denoise"], out0["denoise"] @ R.T, atol=1e-9, rtol=1e-9)
        assert out0["denoise"][1, 4].abs().max() == 0.0
    finally:
        torch.set_default_dtype(prev_dtype)


def test_universe_model_from_mace_synthesizes_2e_tensor_from_0e_1o_backbone():
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(11)
        dummy_mace = _DummyMACE()
        model = YLMUniverseModel.from_mace(
            text_dim=12,
            backbone="mace-mp-0-small",
            mace_model=dummy_mace,
            freeze_mace=True,
            include_mace_forces=True,
            cross_irreps="8x0e + 6x1o + 2x1e + 4x2e",
            n_cross_layers=2,
            option="C",
            c_tp_mode="channelwise",
            d_text=24,
        ).eval()

        Z = torch.tensor([[3, 26, 15, 8]], dtype=torch.long)
        pos = torch.tensor([[[0.0, 0.0, 0.0], [2.8, 0.4, -0.2], [-1.9, 1.7, 0.5], [0.6, -2.1, 1.3]]], dtype=torch.float64)
        mask = torch.ones(1, 4, dtype=torch.bool)
        text = torch.randn(1, 3, 12, dtype=torch.float64)
        tmask = torch.ones(1, 3, dtype=torch.bool)

        with torch.random.fork_rng():
            torch.manual_seed(23)
            R = -o3.rand_matrix().to(torch.float64)

        out0 = model(Z, pos, mask, text, tmask)
        outr = model(Z, pos @ R.T, mask, text, tmask)

        # Verify traceless 2e part of tensor is non-zero (synthesized via 1o x 1o -> 2e CG in Option C!)
        ten0 = out0["tensor"][0]
        traceless0 = ten0 - (ten0.trace() / 3.0) * torch.eye(3, dtype=torch.float64)
        assert traceless0.norm() > 1e-7
        torch.testing.assert_close(outr["tensor"], rotate_matrix(out0["tensor"], R), atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(outr["vector"], out0["vector"] @ R.T, atol=1e-8, rtol=1e-8)
    finally:
        torch.set_default_dtype(prev_dtype)


def test_decoupled_vector_loss_direction_only_and_routing_supervision():
    pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32)
    targ_a = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    targ_b = torch.tensor([[2.0, 0.0, 0.0], [0.0, 9.5, 0.0]], dtype=torch.float32)
    dir_only = torch.tensor([False, True])

    # Row 1 is xtb (direction_only=True): rescaling target magnitude 1.0 -> 9.5 must NOT change loss!
    loss_a = decoupled_vector_loss(pred, targ_a, direction_only=dir_only)
    loss_b = decoupled_vector_loss(pred, targ_b, direction_only=dir_only)
    torch.testing.assert_close(loss_a, loss_b, atol=1e-6, rtol=1e-6)

    # Without direction_only masking, rescaling target magnitude changes loss substantially
    loss_unmasked = decoupled_vector_loss(pred, targ_b, direction_only=None)
    assert float(loss_unmasked) > float(loss_a) + 1.0

    # Routing supervision concentrates attention onto queried target atoms
    logits = torch.zeros(2, 4, requires_grad=True)
    target_atoms = torch.tensor([[False, True, False, False], [True, False, True, False]])
    attn = logits.softmax(dim=-1)
    r_loss = routing_supervision_loss(attn, target_atoms)
    r_loss.backward()
    assert logits.grad[0, 1] < 0.0 and logits.grad[0, 0] > 0.0
    assert logits.grad[1, 0] < 0.0 and logits.grad[1, 2] < 0.0

    # Global queries (no target atoms) produce exact zero routing loss
    empty_loss = routing_supervision_loss(attn.detach(), torch.zeros(2, 4, dtype=torch.bool))
    assert float(empty_loss) == 0.0


def test_fm_baselines_invariance_denoise_and_infonce():
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(5)
        unimol = UniMolStyle3DBaseline(text_dim=12, d_model=32, n_layers=2, n_heads=4, n_kernels=16).eval()
        equillm = EquiLLMMultiTaskBaseline(
            text_dim=12, hidden_irreps="8x0e + 4x1o + 2x2e", n_layers=2, d_model=32, n_vec=8
        ).eval()
        Z = torch.tensor([[6, 8, 17, 1], [3, 8, 6, 9]], dtype=torch.long)
        pos = torch.randn(2, 4, 3, dtype=torch.float64) * 2.0
        mask = torch.ones(2, 4, dtype=torch.bool)
        text = torch.randn(2, 3, 12, dtype=torch.float64)
        tmask = torch.ones(2, 3, dtype=torch.bool)

        with torch.random.fork_rng():
            torch.manual_seed(31)
            R = -o3.rand_matrix().to(torch.float64)

        u0 = unimol(Z, pos, mask, text, tmask)
        ur = unimol(Z, pos @ R.T, mask, text, tmask)
        # Uni-Mol scalar is strictly O(3)-invariant; its internal pair-weighted denoise is O(3)-equivariant!
        torch.testing.assert_close(ur["scalar"], u0["scalar"], atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(ur["denoise"], u0["denoise"] @ R.T, atol=1e-9, rtol=1e-9)
        # Meanwhile Uni-Mol's invariant language vector head CANNOT rotate with R (produces identical vector!)
        torch.testing.assert_close(ur["vector"], u0["vector"], atol=1e-9, rtol=1e-9)

        e0 = equillm(Z, pos, mask, text, tmask)
        er = equillm(Z, pos @ R.T, mask, text, tmask)
        torch.testing.assert_close(er["scalar"], e0["scalar"], atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(er["vector"], e0["vector"] @ R.T, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(er["tensor"], rotate_matrix(e0["tensor"], R), atol=1e-9, rtol=1e-9)
    finally:
        torch.set_default_dtype(prev_dtype)

    # GraphMVP / 3D-InfoMax contrastive InfoNCE check in float32
    torch.manual_seed(9)
    gmvp = GraphMVP3DInfoMaxBaseline(text_dim=12, d_model=32, n_layers=2)
    Z32 = torch.tensor([[6, 8, 1, 1], [15, 8, 9, 17], [3, 8, 6, 8]], dtype=torch.long)
    pos32 = torch.randn(3, 4, 3) * 2.0
    mask32 = torch.ones(3, 4, dtype=torch.bool)
    c_loss = gmvp.contrastive_3d_2d_loss(Z32, pos32, mask32)
    assert torch.isfinite(c_loss) and float(c_loss.detach()) > 0.0
    aligned = torch.eye(4, 8)
    assert float(info_nce_3d_2d_loss(aligned, aligned, temperature=0.1)) < 0.05


def test_universe_data_labels_and_queries_covariant_under_o3():
    frame = StructureFrame(
        Z=[3, 26, 15, 8, 9],
        pos_bohr=np.array([
            [0.6, 0.8, 0.7], [2.4, 2.1, 1.9], [2.5, 0.8, 3.6], [1.3, 0.6, 2.7], [3.6, 1.2, 2.9]
        ]) * ANGSTROM_TO_BOHR,
        cell_bohr=np.array([[5.2, 0.0, 0.0], [0.1, 5.1, 0.0], [0.0, 0.1, 5.3]]) * ANGSTROM_TO_BOHR,
        pbc=np.ones(3, dtype=bool),
        label="pbc_check",
    )
    rng = np.random.default_rng(44)
    q_orth, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R = -q_orth

    frame_r = frame.transformed(R, translation_bohr=np.array([0.4, -0.7, 1.1]))
    lab0 = compute_covariant_universe_labels(frame, lot="pbe_u")
    labr = compute_covariant_universe_labels(frame_r, lot="pbe_u")

    np.testing.assert_allclose(labr["scalar"], lab0["scalar"], atol=1e-9)
    np.testing.assert_allclose(labr["dipole"], lab0["dipole"] @ R.T, atol=1e-9)
    np.testing.assert_allclose(labr["forces"], lab0["forces"] @ R.T, atol=1e-9)
    np.testing.assert_allclose(labr["tensor"], R @ lab0["tensor"] @ R.T, atol=1e-9)

    q0 = generate_diverse_queries(frame, structure_id="s0", domain="crystal_pbc", lot="pbe_u", rng=np.random.default_rng(1))
    qr = generate_diverse_queries(frame_r, structure_id="s0", domain="crystal_pbc", lot="pbe_u", rng=np.random.default_rng(1))
    assert len(q0) == len(qr) and len(q0) >= 5
    for r0, rr in zip(q0, qr):
        assert r0.question == rr.question
        assert r0.target_atoms == rr.target_atoms
        if r0.task_kind == "vector":
            np.testing.assert_allclose(np.asarray(rr.vector_target), np.asarray(r0.vector_target) @ R.T, atol=1e-9)


def test_scripts_train_and_benchmark_cpu_smoke(tmp_path):
    splits = build_cpu_smoke_universe_dataset(n_conformers_per_archetype=3, seed=3)
    featurizer = QuestionFeaturizer(mode="fast", dim=32)
    batch = collate_universe_records(splits["train"][:8], featurizer)
    model = YLMUniverseModel(
        text_dim=32,
        hidden_irreps="8x0e + 4x1o + 2x2e",
        cross_irreps="8x0e + 4x1o + 2x2e",
        n_layers=1,
        n_cross_layers=1,
        d_text=32,
    )
    out = model(
        batch["Z"], batch["pos"], batch["atom_mask"],
        batch["text"], batch["text_mask"],
        lot_id=batch["lot_id"], cell=batch["cell"], pbc=batch["pbc"],
    )
    losses = compute_universe_loss(
        out,
        task_kind=batch["task_kind"],
        scalar_target=batch["scalar_target"],
        vector_target=batch["vector_target"],
        tensor_target=batch["tensor_target"],
        target_atom_mask=batch["target_atom_mask"],
        lot_id=batch["lot_id"],
    )
    d_loss = coordinate_denoise_loss(model, batch["Z"], batch["pos"], batch["atom_mask"], lot_id=batch["lot_id"])
    (losses["loss"] + 0.1 * d_loss).backward()
    metrics = evaluate_universe_o3(model, [batch], n_rotations=2)
    assert metrics["vector_eq_err"] < 1e-4
    assert metrics["tensor_eq_err"] < 1e-4

    # Smoke-test CLI scripts end-to-end in tmp_path
    prep_mod = _load_script("scripts/31_prepare_universe_data.py", "prep_univ")
    data_dir = tmp_path / "u_data"
    manifest = prep_mod.main(["--smoke", "--n_conformers", "3", "--out", str(data_dir)])
    assert manifest["split_counts"]["train"] > 0

    bench_mod = _load_script("scripts/32_benchmark_vs_unimol.py", "bench_univ")
    bench_out = tmp_path / "bench_out"
    payload = bench_mod.main([
        "--data_dir", str(data_dir),
        "--out_dir", str(bench_out),
        "--tiny",
        "--text_dim", "32",
        "--epochs", "1",
        "--batch_size", "16",
        "--eval_rotations", "2",
        "--device", "cpu",
    ])
    assert len(payload["models"]) == 4
    assert (bench_out / "benchmark_table.md").exists()

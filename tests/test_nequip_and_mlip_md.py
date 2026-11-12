"""Tests for NequIP/Allegro backbone adapter (`src/nequip_backbone.py`) and MLIP MD tools (`src/toolkit/mlip_md.py`)."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.model import YLM
from src.nequip_backbone import NequIPEncoder, load_nequip_model
from src.toolkit.cli import main as cli_main
from src.toolkit.ingest import ANGSTROM_TO_BOHR, StructureFrame
from src.toolkit.mlip_md import (
    AnalyticalReactiveForceProvider,
    MLIPTrajectoryRunner,
    TorchMLIPForceProvider,
    build_openmm_mlip_simulation,
)
from src.toolkit.probe import YLMProbe
from src.universe_fm import YLMUniverseModel
from src.ylm_lm import YLMCausalLM, YLMConfig


class _DummyNequIPGraphModule(nn.Module):
    """Exact O(3)-equivariant NequIP-style GraphModule outputting `node_features` (4x0e + 2x1o) + `total_energy`."""

    def __init__(self):
        super().__init__()
        self.r_max = 5.5
        self.type_names = ["H", "C", "N", "O", "F", "Li"]
        self.node_features_irreps = "4x0e + 2x1o"
        self.species_emb = nn.Embedding(len(self.type_names), 4)
        self.vec_weights = nn.Linear(4, 2, bias=False)

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pos = data["pos"]
        atom_types = data["atom_types"]
        edge_index = data["edge_index"]
        shifts = data["edge_cell_shift"]
        cell = data["cell"]
        batch = data.get("batch", torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device))

        n_atoms = pos.shape[0]
        h0 = self.species_emb(atom_types)  # (M, 4) invariant 0e

        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            if cell.ndim == 3:
                shift_cart = torch.einsum("ei,eij->ej", shifts, cell[batch[src]])
            else:
                shift_cart = shifts @ cell
            rel = pos[dst] - pos[src] + shift_cart  # (E, 3) 1o
            dist2 = (rel * rel).sum(dim=-1, keepdim=True)
            env = torch.exp(-0.35 * dist2)
            pair_e = 18.0 * env * (h0[src, :1].square() + h0[dst, :1].square() + 0.25)

            atom_e = torch.zeros((n_atoms, 1), dtype=pos.dtype, device=pos.device)
            atom_e.index_add_(0, dst, 0.5 * pair_e)

            # Aggregate 2x1o equivariant node vectors
            w_pair = self.vec_weights(h0[src]) * env  # (E, 2)
            v_msg = w_pair[:, :, None] * rel[:, None, :]  # (E, 2, 3)
            v_agg = torch.zeros((n_atoms, 2, 3), dtype=pos.dtype, device=pos.device)
            v_agg.index_add_(0, dst, v_msg)

            s_agg = torch.zeros_like(h0)
            s_agg.index_add_(0, dst, h0[src] * env)
            scalars = h0 + s_agg
        else:
            atom_e = h0[:, :1] * 0.0
            v_agg = torch.zeros((n_atoms, 2, 3), dtype=pos.dtype, device=pos.device)
            scalars = h0

        node_features = torch.cat([scalars, v_agg.reshape(n_atoms, 6)], dim=-1)
        total_energy = atom_e.sum(dim=0)
        return {
            "node_features": node_features,
            "atomic_energy": atom_e,
            "total_energy": total_energy,
        }


class _ScriptableDeployedNequIP(nn.Module):
    """TorchScript-compilable `nequip-deploy` energy+force module (`cell` shape `(3, 3)`)."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([1.0, 1.4, 1.8, 2.2], dtype=torch.float64))

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pos = data["pos"]
        atom_types = data["atom_types"]
        edge_index = data["edge_index"]
        shifts = data["edge_cell_shift"]
        cell = data["cell"]

        n_atoms = pos.shape[0]
        w = self.scale[atom_types]
        atom_e = torch.zeros((n_atoms, 1), dtype=pos.dtype, device=pos.device)
        if edge_index.shape[1] > 0:
            src = edge_index[0]
            dst = edge_index[1]
            if cell.dim() == 2:
                shift_cart = torch.matmul(shifts, cell)
            else:
                shift_cart = torch.zeros_like(shifts)
            rel = pos[dst] - pos[src] + shift_cart
            d2 = (rel * rel).sum(dim=-1, keepdim=True)
            e_ij = 8.0 * torch.exp(-0.3 * d2) * (w[src] * w[dst]).unsqueeze(-1)
            atom_e.index_add_(0, dst, 0.5 * e_ij)
        total_e = atom_e.sum()
        grads = torch.autograd.grad([total_e], [pos], create_graph=False)[0]
        assert grads is not None
        return {
            "atomic_energy": atom_e,
            "total_energy": total_e.unsqueeze(0),
            "forces": -grads,
        }


def _random_improper_o3(seed: int = 19) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=gen, dtype=torch.float64))
    d = torch.diagonal(r)
    q = q * torch.where(d >= 0, torch.ones_like(d), -torch.ones_like(d))
    if torch.linalg.det(q) > 0:
        q[:, 0] = -q[:, 0]  # improper O(3) reflection+rotation (det = -1)
    return q


def test_nequip_encoder_molecule_and_pbc_exact_o3_equivariance():
    torch.manual_seed(7)
    nq = _DummyNequIPGraphModule().double()
    enc = NequIPEncoder(backbone="nequip:mock", nequip_model=nq, include_nequip_forces=True).double().eval()
    assert str(enc.irreps_out) == "4x0e+2x1o+1x1o"

    # Periodic crystal test (Li-C-O-H supercell in Bohr)
    Z = torch.tensor([[3, 6, 8, 1]], dtype=torch.long)
    pos = torch.tensor(
        [[[0.1, 0.2, 0.3], [2.4, 0.5, -0.2], [1.9, 2.2, 0.4], [0.8, 1.5, 1.9]]],
        dtype=torch.float64,
    )
    mask = torch.ones((1, 4), dtype=torch.bool)
    cell = torch.tensor([[[7.5, 0.4, 0.0], [0.0, 7.8, 0.3], [0.2, 0.0, 8.1]]], dtype=torch.float64)
    pbc = torch.tensor([[True, True, True]])

    R = _random_improper_o3(seed=31)
    D_mat = enc.irreps_out.D_from_matrix(R)

    y0 = enc(Z, pos, mask, cell=cell, pbc=pbc)
    y_rot = enc(Z, pos @ R.T, mask, cell=cell @ R.T, pbc=pbc)
    y_expected = y0 @ D_mat.T
    rel_err = (y_rot - y_expected).norm() / y0.norm().clamp(min=1e-12)
    assert float(rel_err) < 1e-10


def test_torchscript_deployed_nequip_file_and_ylm_integrations(tmp_path: Path):
    torch.manual_seed(11)
    scripted = torch.jit.script(_ScriptableDeployedNequIP().double())
    ckpt_path = tmp_path / "allegro_deployed.pth"
    extra = {
        "r_max": b"5.0",
        "type_names": b"H C N O",
        "nequip_version": b"0.6.1",
    }
    torch.jit.save(scripted, str(ckpt_path), _extra_files=extra)

    loaded_model, meta = load_nequip_model(f"allegro:{ckpt_path}")
    assert meta["type_names"] == "H C N O"
    assert float(meta["r_max"]) == pytest.approx(5.0)
    assert isinstance(loaded_model, torch.jit.ScriptModule)

    # Mount into YLM, YLMUniverseModel, YLMCausalLM, and YLMProbe
    ylm = YLM.from_nequip(text_dim=32, backbone=f"nequip:{ckpt_path}", option="C", c_tp_mode="channelwise").double().eval()
    Z = torch.tensor([[8, 1, 1]], dtype=torch.long)
    pos = torch.tensor([[[0.0, 0.0, 0.2], [0.0, 1.4, -0.8], [0.0, -1.4, -0.8]]], dtype=torch.float64)
    amask = torch.ones((1, 3), dtype=torch.bool)
    txt = torch.randn(1, 5, 32, dtype=torch.float64)
    tmask = torch.ones((1, 5), dtype=torch.bool)
    task = torch.tensor([1], dtype=torch.long)

    R = _random_improper_o3(seed=43)
    v0, a0 = ylm(Z, pos, amask, txt, tmask, task)
    vr, ar = ylm(Z, pos @ R.T, amask, txt, tmask, task)
    assert float((vr - v0 @ R.T).norm()) < 1e-10
    assert float((ar - a0).abs().max()) < 1e-10

    uni = YLMUniverseModel.from_nequip(text_dim=32, backbone=f"nequip:{ckpt_path}").double().eval()
    u0 = uni(Z, pos, amask, txt, tmask)
    ur = uni(Z, pos @ R.T, amask, txt, tmask)
    assert float((ur["scalar"] - u0["scalar"]).abs().max()) < 1e-10
    assert float((ur["vector"] - u0["vector"] @ R.T).norm()) < 1e-10
    assert float((ur["tensor"] - (R @ u0["tensor"] @ R.T)).norm()) < 1e-10

    # YLMCausalLM with NequIP backbone
    cfg = YLMConfig(mode="xattnA", lm="tiny", vocab_size=64, encoder=f"nequip:{ckpt_path}", vec_readout=True, vec_token_id=5)
    lm = YLMCausalLM(cfg).eval()
    ids = torch.tensor([[1, 10, 5, 2]], dtype=torch.long)
    attn_m = torch.ones_like(ids)
    geom = {
        "Z": Z,
        "pos": pos.float(),
        "atom_mask": amask,
        "role": torch.zeros_like(Z),
        "amount": torch.ones((1, 3), dtype=torch.float32),
    }
    out_lm = lm(ids, attn_m, geom=geom)
    assert out_lm["vec"].shape == (1, 3)

    # YLMProbe + heavy-element valence fallback check (Z=55 Cs -> H)
    probe = YLMProbe.from_nequip(backbone=f"nequip:{ckpt_path}")
    chk = probe.verify_equivariance("O", "What is the force vector on the oxygen atom?", tol=1e-4)
    assert chk.passed

    cs_frame = StructureFrame(Z=[55, 8], pos_bohr=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]))
    with pytest.warns(UserWarning, match="Z=55"):
        res_cs = probe.ask(cs_frame, "What is the force on oxygen?")
    assert any("Z=55" in w for w in res_cs.warnings)


def test_analytical_and_nequip_force_providers_and_mlip_md_runner(tmp_path: Path):
    # 1) Verify exact finite-difference conservation & O(3) covariance of AnalyticalReactiveForceProvider
    ana = AnalyticalReactiveForceProvider()
    z = np.array([8, 1, 8], dtype=np.int64)
    pos = np.array([[0.0, 0.0, 0.0], [1.05, 0.05, 0.0], [2.45, 0.0, 0.0]], dtype=np.float64)
    e0, f0 = ana(z, pos)
    eps = 1e-6
    pos_plus = pos.copy()
    pos_plus[1, 0] += eps
    pos_minus = pos.copy()
    pos_minus[1, 0] -= eps
    ep, _ = ana(z, pos_plus)
    em, _ = ana(z, pos_minus)
    fd_fx = -(ep - em) / (2.0 * eps)
    assert f0[1, 0] == pytest.approx(fd_fx, rel=1e-5, abs=1e-6)

    R = _random_improper_o3(seed=99).numpy()
    er, fr = ana(z, pos @ R.T)
    assert er == pytest.approx(e0, abs=1e-10)
    np.testing.assert_allclose(fr, f0 @ R.T, atol=1e-10)

    # 2) Drive reactive proton-transfer MD with MLIPTrajectoryRunner (high initial H velocity toward acceptor O#2)
    frame_pt = StructureFrame(Z=z, pos_bohr=pos * ANGSTROM_TO_BOHR, label="proton_hop")
    v_init = np.array([[0.0, 0.0, 0.0], [0.16, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    runner = MLIPTrajectoryRunner(ana, dt_fs=0.5, temperature_k=0.0, friction_per_fs=0.0)

    traj_p = tmp_path / "md.extxyz"
    json_p = tmp_path / "events.json"
    svg_p = tmp_path / "timeline.svg"
    html_p = tmp_path / "traj.html"
    md_res = runner.run(
        frame_pt,
        steps=14,
        initial_velocities_ang_fs=v_init,
        out_extxyz=traj_p,
        out_events_json=json_p,
        out_plot=svg_p,
        out_html=html_p,
    )
    assert len(md_res.frames) == 15
    assert any(ev.event_type == "proton_transfer" for ev in md_res.events)
    assert traj_p.exists() and svg_p.exists() and html_p.exists()
    ev_data = json.loads(json_p.read_text(encoding="utf-8"))
    assert ev_data["n_events"] >= 1

    # 3) Test NequIPEncoder force provider through TorchMLIPForceProvider
    nq_enc = NequIPEncoder(backbone="nequip:mock", nequip_model=_DummyNequIPGraphModule())
    nq_provider = TorchMLIPForceProvider(nq_enc)
    e_nq, f_nq = nq_provider(z, pos)
    assert np.isfinite(e_nq) and f_nq.shape == (3, 3)

    # 4) Test CLI `ylm run-md`
    cli_traj = tmp_path / "cli_md.extxyz"
    cli_json = tmp_path / "cli_events.json"
    rc = cli_main([
        "run-md",
        "O",
        "--steps", "4",
        "--out-traj", str(cli_traj),
        "--json", str(cli_json),
    ])
    assert rc == 0
    assert cli_traj.exists() and cli_json.exists()

    # 5) Verify helpful ImportError when OpenMM is absent
    with pytest.raises(ImportError, match="OpenMM"):
        build_openmm_mlip_simulation(None, None)

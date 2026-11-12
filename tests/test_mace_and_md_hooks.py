"""Tests for MACEEncoder (in YLM, YLMCausalLM, and YLMProbe) and OpenMM/LAMMPS live MD hooks."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from e3nn import o3

from src.encoder import bessel_basis
from src.mace_backbone import MACEEncoder
from src.model import YLM
from src.tensor_readout import rotate_matrix
from src.toolkit.cli import main as cli_main
from src.toolkit.md_hooks import (
    LAMMPSReactionHook,
    LiveReactionMonitor,
    OpenMMReactionReporter,
    plot_reaction_timeline,
)
from src.toolkit.probe import YLMProbe
from src.ylm_lm import YLMConfig, YLMCausalLM, collate_geometry


class _ProductStub(nn.Module):
    def __init__(self, irreps_in: str, irreps_out: str):
        super().__init__()
        self.linear = o3.Linear(irreps_in, irreps_out)


class SyntheticMACEModule(nn.Module):
    """Genuine O(3)-equivariant e3nn MACE-compatible module for fast deterministic offline tests.

    Follows MACE's exact forward contract:
      inputs: `data["positions"]` (Å), `node_attrs`, `edge_index` `[src, dst]`, `shifts` (Å), `batch`, `ptr`
      attributes: `.atomic_numbers`, `.r_max`, `.products[i].linear.irreps_out`
      outputs: `{"node_feats": ..., "energy": ..., "forces": -dE/d(positions)}`
    """

    def __init__(self, atomic_numbers=(1, 3, 6, 7, 8, 9, 15, 16, 17, 26), r_max: float = 5.0):
        super().__init__()
        self.register_buffer("atomic_numbers", torch.tensor(list(atomic_numbers), dtype=torch.long))
        self.register_buffer("r_max", torch.tensor(float(r_max)))
        n_sp = len(atomic_numbers)
        self.irreps_sh = o3.Irreps.spherical_harmonics(2)
        self.node_embed = nn.Linear(n_sp, 8, bias=False)
        self.tp = o3.FullyConnectedTensorProduct("8x0e", self.irreps_sh, "8x0e + 6x1o + 4x2e", shared_weights=False)
        self.rad_net = nn.Sequential(nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, self.tp.weight_numel))
        self.products = nn.ModuleList([
            _ProductStub("8x0e + 6x1o + 4x2e", "8x0e + 6x1o + 4x2e"),
            _ProductStub("8x0e + 6x1o + 4x2e", "8x0e"),
        ])
        self.energy_head = nn.Linear(8, 1, bias=False)

    def forward(self, data: dict, training: bool = False, compute_force: bool = True) -> dict:
        positions = data["positions"]
        node_attrs = data["node_attrs"]
        src, dst = data["edge_index"][0], data["edge_index"][1]
        shifts = data["shifts"]
        n_nodes = positions.shape[0]

        h0 = self.node_embed(node_attrs)  # (M, 8)
        if src.numel() > 0:
            # MACE edge vector convention: r_dst - r_src + shifts
            vectors = positions[dst] - positions[src] + shifts
            lengths = vectors.norm(dim=-1)
            sh = o3.spherical_harmonics(self.irreps_sh, vectors, normalize=True, normalization="component")
            rad = bessel_basis(lengths, float(self.r_max.item()), 6)
            msg = self.tp(h0[src], sh, self.rad_net(rad))
            agg = torch.zeros((n_nodes, msg.shape[-1]), dtype=positions.dtype, device=positions.device)
            agg.index_add_(0, dst, msg / 6.0)
        else:
            agg = torch.zeros((n_nodes, self.tp.irreps_out.dim), dtype=positions.dtype, device=positions.device)

        f1 = self.products[0].linear(agg)
        f2 = self.products[1].linear(agg)
        node_feats = torch.cat([f1, f2], dim=-1)

        # Add smooth pairwise harmonic interaction energy + learned scalar readout for exact conservative forces
        pair_e = (lengths - 1.4).pow(2).sum() * 0.15 if src.numel() > 0 else positions.sum() * 0.0
        total_e = self.energy_head(f2).sum() + pair_e

        forces = None
        if compute_force:
            grad = torch.autograd.grad(
                outputs=[total_e],
                inputs=[positions],
                create_graph=training,
                retain_graph=training,
                allow_unused=True,
            )[0]
            forces = -grad if grad is not None else torch.zeros_like(positions)

        return {
            "node_feats": node_feats,
            "energy": total_e,
            "forces": forces,
        }


def _rand_o3(seed: int, improper: bool = False) -> torch.Tensor:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        r = o3.rand_matrix(dtype=torch.float64)
    return -r if improper else r


@pytest.mark.parametrize("improper", [False, True])
@pytest.mark.parametrize("option", ["A", "B", "C"])
def test_ylm_with_mace_backbone_o3_equivariance_and_heavy_elements(improper: bool, option: str):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(11)
        mace_mod = SyntheticMACEModule()
        model = YLM.from_mace(
            text_dim=16,
            backbone="mace-mp-0-small",
            mace_model=mace_mod,
            freeze_mace=True,
            option=option,
            c_tp_mode="channelwise",
            cross_irreps="6x0e + 4x1o + 2x1e + 2x2e",
        )
        # Include heavy elements: Fe (26), P (15), O (8), Li (3)
        Z = torch.tensor([[26, 15, 8, 3], [6, 17, 1, 0]], dtype=torch.long)
        pos = torch.randn(2, 4, 3, dtype=torch.float64) * 2.2
        mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
        text = torch.randn(2, 3, 16, dtype=torch.float64)
        text_mask = torch.tensor([[True, True, True], [True, True, False]])
        task = torch.tensor([1, 0], dtype=torch.long)

        R = _rand_o3(seed=17, improper=improper)
        # Verify inference inside torch.no_grad() still computes conservative MACE forces cleanly
        with torch.no_grad():
            v0, a0 = model(Z, pos, mask, text, text_mask, task)
            vr, ar = model(Z, pos @ R.T, mask, text, text_mask, task)

        torch.testing.assert_close(vr, v0 @ R.T, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(ar, a0, atol=1e-9, rtol=1e-9)

        # Also check periodic crystal cell + tensor_readout=True (0e+2e symmetric matrix)
        model_quad = YLM.from_mace(
            text_dim=16,
            backbone="mace-mp-0-small",
            mace_model=mace_mod,
            option="C",
            c_tp_mode="channelwise",
            cross_irreps="6x0e + 4x1o + 2x1e + 2x2e",
            tensor_readout=True,
        )
        cell = torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(2, 3, 3) * 8.5
        pbc = torch.ones((2, 3), dtype=torch.bool)
        with torch.no_grad():
            q0, _ = model_quad(Z, pos, mask, text, text_mask, task, cell=cell, pbc=pbc)
            qr, _ = model_quad(Z, pos @ R.T, mask, text, text_mask, task, cell=cell @ R.T, pbc=pbc)
        torch.testing.assert_close(qr, rotate_matrix(q0, R), atol=1e-8, rtol=1e-8)

        # Check gradient isolation when freeze_mace=True
        pred, _ = model(Z, pos, mask, text, text_mask, task)
        pred.pow(2).sum().backward()
        assert model.readout.weight.grad is not None and model.readout.weight.grad.abs().sum() > 0
        for p in mace_mod.parameters():
            assert p.grad is None
    finally:
        torch.set_default_dtype(prev_dtype)


def test_ylm_causal_lm_and_probe_with_mace_backbone():
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(23)
        mace_mod = SyntheticMACEModule(atomic_numbers=(1, 6, 7, 8, 9, 16, 17))  # organic subset like MACE-OFF23
        cfg = YLMConfig(
            mode="xattnC",
            c_tp_mode="channelwise",
            lm="tiny",
            vocab_size=32,
            n_layer=2,
            n_embd=32,
            n_head=4,
            encoder="mace-off23-small",
            cross_irreps="4x0e + 4x1o + 2x1e + 2x2e",
            vec_readout=True,
            vec_token_id=4,
            use_motion_1o=True,
            freeze_encoder=True,
        )
        lm = YLMCausalLM(cfg, mace_model=mace_mod).eval()
        # Activate adapter gate so geometry flows into LM logits as well as <vec> stream
        with torch.no_grad():
            lm.adapters[0].gate.fill_(0.7)

        pos0 = np.array([[0.0, 0.0, 0.0], [2.2, 0.3, -0.1], [-1.5, 1.8, 0.4]], dtype=np.float64)
        dpos0 = np.array([[0.05, -0.02, 0.01], [-0.03, 0.04, 0.0], [0.01, -0.01, -0.02]], dtype=np.float64)
        geom0 = collate_geometry([{"Z": [6, 8, 1], "pos": pos0, "delta_pos": dpos0}], dtype=torch.float64)
        input_ids = torch.tensor([[1, 7, 4, 9, 2]], dtype=torch.long)
        attn_mask = torch.ones_like(input_ids)

        R = _rand_o3(seed=29, improper=True)
        r_np = R.numpy()
        geom_r = collate_geometry([{"Z": [6, 8, 1], "pos": pos0 @ r_np.T, "delta_pos": dpos0 @ r_np.T}], dtype=torch.float64)

        with torch.no_grad():
            out0 = lm(input_ids, attn_mask, geom=geom0)
            outr = lm(input_ids, attn_mask, geom=geom_r)

        torch.testing.assert_close(outr["logits"], out0["logits"], atol=1e-9, rtol=1e-9)
        v0 = out0["vec"][0]
        vr = outr["vec"][0]
        torch.testing.assert_close(vr, v0 @ R.T, atol=1e-9, rtol=1e-9)

        # Also check YLMProbe.from_mace + out-of-vocabulary element fallback warning (Li=3 on MACE-OFF23 subset)
        torch.set_default_dtype(torch.float32)
        mace_f32 = SyntheticMACEModule(atomic_numbers=(1, 6, 7, 8, 9, 16, 17)).float()
        probe = YLMProbe.from_mace("mace-off23-small", mace_model=mace_f32, text_dim=32)
        with pytest.warns(UserWarning, match="Z=3"):
            res = probe.ask("examples/li_ec_shell.xyz", "What is the force on the lithium ion?", verify_o3=True)
        assert res.o3_check is not None and res.o3_check.passed
        assert any("Z=3" in w for w in res.warnings)
    finally:
        torch.set_default_dtype(prev_dtype)


def test_live_md_hooks_openmm_lammps_and_timeline_plot(tmp_path):
    # 1) Proton transfer trajectory via CLI --plot
    svg_cli = tmp_path / "grotthuss_timeline.svg"
    rc = cli_main(["scan-traj", "examples/proton_transfer_traj.extxyz", "--dt-fs", "2.5", "--plot", str(svg_cli)])
    assert rc == 0
    assert svg_cli.exists()
    svg_text = svg_cli.read_text(encoding="utf-8")
    assert "proton_transfer" in svg_text

    # 2) OpenMMReactionReporter duck-typed simulation (bond breaking + bond forming)
    omm_json = tmp_path / "openmm_events.json"
    omm_svg = tmp_path / "openmm_timeline.svg"
    reporter = OpenMMReactionReporter(omm_json, reportInterval=10, plot_path=omm_svg, atomic_numbers=[6, 8, 1])
    sim_mock = SimpleNamespace(currentStep=3)
    assert reporter.describeNextReport(sim_mock) == (7, True, False, True, False, True)

    frames_pos = [
        np.array([[0.0, 0.0, 0.0], [1.35, 0.0, 0.0], [4.0, 0.0, 0.0]]),   # t=0 fs: C-O bonded, H far
        np.array([[0.0, 0.0, 0.0], [2.60, 0.0, 0.0], [2.60, 0.95, 0.0]]), # t=5 fs: C-O broken, O-H formed!
    ]
    forces = np.array([[-0.5, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, -0.4, 0.0]])
    for step_i, pos_ang in enumerate(frames_pos):
        st = SimpleNamespace(
            getPositions=lambda p=pos_ang: p,
            getForces=lambda: forces,
            getTime=lambda s=step_i: s * 5.0,
        )
        reporter.report(sim_mock, st)

    assert omm_json.exists() and omm_svg.exists()
    saved = json.loads(omm_json.read_text(encoding="utf-8"))
    ev_types = {e["event_type"] for e in saved["events"]}
    assert "bond_broken" in ev_types
    assert "bond_formed" in ev_types

    # 3) LAMMPSReactionHook callback
    lmp_json = tmp_path / "lammps_events.json"
    lmp_svg = tmp_path / "lammps_timeline.svg"
    hook = LAMMPSReactionHook({1: 8, 2: 1}, lmp_json, dt_fs=0.5, plot_path=lmp_svg)
    lmp_frames = [
        (0, [1, 2, 1], np.array([[0.0, 0.0, 0.0], [0.98, 0.0, 0.0], [2.50, 0.0, 0.0]]), np.zeros((3, 3))),
        (20, [1, 2, 1], np.array([[0.0, 0.0, 0.0], [1.52, 0.0, 0.0], [2.50, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 0.0, 0.0]])),
    ]
    for nts, types, x_arr, f_arr in lmp_frames:
        ptr_mock = SimpleNamespace(extract_atom_numpy=lambda a=nts, b=types, c=x_arr, d=f_arr: (a, b, c, d))
        hook(ptr_mock)

    assert lmp_json.exists() and lmp_svg.exists()
    lmp_saved = json.loads(lmp_json.read_text(encoding="utf-8"))
    assert any(e["event_type"] == "proton_transfer" and abs(e["time_fs"] - 10.0) < 1e-6 for e in lmp_saved["events"])
    out_plot = plot_reaction_timeline(lmp_saved["history"], hook.monitor.events, tmp_path / "check.svg")
    assert out_plot.exists()

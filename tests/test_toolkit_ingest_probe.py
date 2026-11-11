"""Tests for universal structure/trajectory ingestion, 1o motion equivariance, and YLMProbe."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch
from e3nn import o3

from src.tasks.common import Example, Seg, collate, model_inputs
from src.tasks.tokenizer import CharTokenizer, VEC_TOKEN
from src.toolkit.ingest import (
    ANGSTROM_TO_BOHR,
    StructureFrame,
    build_solvation_cluster,
    load_structure,
    load_trajectory,
    minimum_image_delta_bohr,
    smiles_to_frame,
)
from src.toolkit.probe import YLMProbe, verify_o3_equivariance
from src.ylm_lm import YLMCausalLM, YLMConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


SAMPLE_SDF_V2000 = """water_v2000
  YLM-Test          3D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.1173 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
"""

SAMPLE_POSCAR = """Cubic LiF
1.0
4.02 0.00 0.00
0.00 4.02 0.00
0.00 0.00 4.02
Li F
1 1
Direct
0.00 0.00 0.00
0.50 0.50 0.50
"""

SAMPLE_CIF = """data_LiH
_cell_length_a 4.08
_cell_length_b 4.08
_cell_length_c 4.08
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li 0.0 0.0 0.0
H  0.5 0.5 0.5
"""


def test_ingest_formats_and_periodic_minimum_image(tmp_path: Path):
    f_xyz = load_structure(REPO_ROOT / "examples" / "water.xyz")
    assert f_xyz.n_atoms == 3
    assert f_xyz.symbols == ["O", "H", "H"]

    sdf_file = tmp_path / "water.sdf"
    sdf_file.write_text(SAMPLE_SDF_V2000, encoding="utf-8")
    f_sdf = load_structure(sdf_file)
    assert f_sdf.n_atoms == 3
    assert f_sdf.symbols[0] == "O"

    poscar_file = tmp_path / "POSCAR"
    poscar_file.write_text(SAMPLE_POSCAR, encoding="utf-8")
    f_poscar = load_structure(poscar_file)
    assert f_poscar.n_atoms == 2
    assert f_poscar.cell_bohr is not None
    assert bool(f_poscar.pbc.all())

    cif_file = tmp_path / "lih.cif"
    cif_file.write_text(SAMPLE_CIF, encoding="utf-8")
    f_cif = load_structure(cif_file)
    assert f_cif.n_atoms == 2
    assert f_cif.cell_bohr is not None

    f_smi = smiles_to_frame("O=C1OCCO1")
    assert f_smi.n_atoms == 10

    cluster = build_solvation_cluster(["O=C1OCCO1", "O=C1OCCO1", "O"], ion_symbol="Li")
    assert cluster.symbols[0] == "Li"
    assert int(cluster.role[0]) == 1
    assert cluster.n_atoms == 1 + 10 + 10 + 3

    traj = load_trajectory(REPO_ROOT / "examples" / "proton_transfer_traj.extxyz")
    assert len(traj) == 3
    assert traj[1].delta_pos_bohr is not None
    assert np.linalg.norm(traj[1].delta_pos_angstrom) > 0.05

    # Verify minimum-image wrapping across a periodic boundary
    cell = np.eye(3) * 10.0
    p0 = np.array([[0.2, 1.0, 1.0]])
    p1 = np.array([[9.9, 1.0, 1.0]])  # crossed left boundary by -0.3 bohr
    delta = minimum_image_delta_bohr(p1, p0, cell, np.ones(3, dtype=bool))
    np.testing.assert_allclose(delta, [[-0.3, 0.0, 0.0]], atol=1e-12)


def test_motion_1o_ylm_causal_lm_exact_float64_o3_equivariance():
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        tok = CharTokenizer()
        cfg = YLMConfig(
            mode="xattnC",
            lm="tiny",
            vocab_size=tok.vocab_size,
            bos_token_id=tok.bos_id,
            eos_token_id=tok.eos_id,
            n_layer=2,
            n_embd=64,
            n_head=4,
            c_tp_mode="channelwise",
            vec_readout=True,
            vec_token_id=tok.vec_id,
            use_motion_1o=True,
        )
        model = YLMCausalLM(cfg).eval()
        for adapter in model.adapters:
            adapter.gate.data.fill_(0.45)

        frame = StructureFrame(
            Z=[8, 1, 1, 6],
            pos_bohr=np.array([[0.0, 0.0, 0.1], [1.4, 0.2, -0.6], [-1.3, 0.4, -0.5], [0.2, -2.1, 0.3]]),
            cell_bohr=np.eye(3) * 14.0,
            pbc=np.array([True, True, True]),
            delta_pos_bohr=np.array([[0.02, -0.01, 0.03], [-0.18, 0.05, 0.11], [0.04, -0.07, 0.02], [0.01, 0.03, -0.02]]),
        )

        with torch.random.fork_rng():
            torch.manual_seed(42)
            R_proper = o3.rand_matrix(dtype=torch.float64).numpy()
            R_improper = -o3.rand_matrix(dtype=torch.float64).numpy()  # det(R) = -1 parity reflection + rotation
        shift = np.array([2.7, -1.3, 0.9])

        prompt = f"Force along motion? {VEC_TOKEN}"
        b0 = collate([Example([Seg(prompt)], frame.to_geom_dict())], tok, max_len=64, dtype=torch.float64)
        out0 = model(**model_inputs(b0), output_routing=True)

        for R in (R_proper, R_improper):
            moved = frame.transformed(R, shift)
            b_rot = collate([Example([Seg(prompt)], moved.to_geom_dict())], tok, max_len=64, dtype=torch.float64)
            out_rot = model(**model_inputs(b_rot), output_routing=True)

            logit_diff = (out0["logits"] - out_rot["logits"]).abs().max().item()
            attn_diff = (out0["atom_attention"] - out_rot["atom_attention"]).abs().max().item()
            assert logit_diff < 1e-10, f"Logits broke O(3) invariance: {logit_diff}"
            assert attn_diff < 1e-10, f"Atom attention broke O(3) invariance: {attn_diff}"

            v0 = out0["vec"].detach().numpy()
            v_rot = out_rot["vec"].detach().numpy()
            np.testing.assert_allclose(v_rot, v0 @ R.T, atol=1e-10, rtol=1e-10)
    finally:
        torch.set_default_dtype(prev_dtype)


def test_ylm_probe_v0_and_causal_lm_and_heavy_element_warning():
    probe = YLMProbe.default()
    res = probe.ask(
        REPO_ROOT / "examples" / "water.xyz",
        "What is the force vector on the oxygen atom?",
        verify_o3=True,
    )
    assert res.engine == "v0"
    assert len(res.vectors) == 1
    assert res.vectors[0].unit == "eV/A"
    assert res.o3_check is not None and res.o3_check.passed
    assert sum(res.l_channel_weights.values()) == pytest.approx(1.0, abs=1e-5)
    assert len(res.top_attended_atoms) == 3

    # Heavy element Z >= 10 triggers explicit valence-analogue warning on QM9 v0 checkpoint
    heavy = StructureFrame(
        Z=[11, 17],  # NaCl
        pos_bohr=np.array([[0.0, 0.0, 0.0], [2.36 * ANGSTROM_TO_BOHR, 0.0, 0.0]]),
    )
    res_heavy = probe.ask(heavy, "What is the dipole moment of this molecule?")
    assert any("remapped heavy atoms" in w for w in res_heavy.warnings)
    assert res_heavy.vectors[0].unit == "Debye"

    chk = verify_o3_equivariance(REPO_ROOT / "examples" / "water.xyz", "What is the dipole moment?")
    assert chk.passed and chk.reflection_tested

    # Test YLMProbe wrapping an in-memory YLMCausalLM with <vec> readout
    tok = CharTokenizer()
    cfg = YLMConfig(
        mode="xattnC",
        lm="tiny",
        vocab_size=tok.vocab_size,
        bos_token_id=tok.bos_id,
        eos_token_id=tok.eos_id,
        n_layer=2,
        n_embd=48,
        n_head=4,
        c_tp_mode="channelwise",
        vec_readout=True,
        vec_token_id=tok.vec_id,
    )
    lm_model = YLMCausalLM(cfg).eval()
    lm_probe = YLMProbe.from_causal_lm(lm_model, tokenizer=tok)
    res_lm = lm_probe.ask(REPO_ROOT / "examples" / "water.xyz", "Force on O?", max_new=6, verify_o3=True)
    assert res_lm.engine == "causal_lm"
    assert len(res_lm.vectors) >= 1
    assert res_lm.o3_check is not None and res_lm.o3_check.passed

"""Tests for periodic boundary condition crystal graphs and pyproject.toml packaging."""
from pathlib import Path
import json
import sys
import tomllib
import types

import numpy as np
import pytest
import torch
from e3nn import o3

from src.encoder import Encoder
from src.ylm_lm import YLMConfig, YLMCausalLM, collate_geometry
from src.tasks import mp_battery as mp
from src.tasks.common import Example, Seg, collate, model_inputs, seed_everything
from src.tasks.conformers import ANGSTROM_TO_BOHR, single_geometry
from src.tasks.geometry_controls import geometry_control
from src.tasks.tokenizer import CharTokenizer


def make_triclinic_crystal():
    # Skewed triclinic cell in bohr small enough that periodic & self-images lie inside cutoff=6.5.
    cell = torch.tensor([
        [4.8, 0.0, 0.0],
        [1.1, 4.6, 0.0],
        [0.6, 0.8, 5.0],
    ], dtype=torch.float64)
    frac = torch.tensor([
        [0.10, 0.20, 0.15],
        [0.55, 0.65, 0.40],
        [0.80, 0.30, 0.75],
    ], dtype=torch.float64)
    pos = frac @ cell
    Z = torch.tensor([[3, 26, 8]], dtype=torch.long)
    mask = torch.ones(1, 3, dtype=torch.bool)
    return Z, pos.unsqueeze(0), mask, cell.unsqueeze(0)


def test_pbc_matches_nonperiodic_for_large_box():
    torch.manual_seed(0)
    enc = Encoder(hidden_irreps="8x0e + 4x1o + 2x2e", n_layers=2, cutoff=6.5, n_elements=30).double().eval()
    Z = torch.tensor([[8, 1, 1]], dtype=torch.long)
    pos = torch.tensor([[[5.0, 5.0, 5.0], [6.4, 5.2, 5.0], [4.6, 6.3, 5.1]]], dtype=torch.float64)
    mask = torch.ones(1, 3, dtype=torch.bool)
    huge_cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 40.0

    out_mol = enc(Z, pos, mask)
    out_pbc = enc(Z, pos, mask, cell=huge_cell, pbc=torch.ones(1, 3, dtype=torch.bool))
    torch.testing.assert_close(out_pbc, out_mol, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("reflection", [False, True])
def test_pbc_o3_equivariance_and_lattice_shift_invariance(reflection):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(4)
        enc = Encoder(hidden_irreps="8x0e + 2x0o + 4x1o + 2x1e + 2x2e", n_layers=2, cutoff=6.5, n_elements=30).eval()
        Z, pos, mask, cell = make_triclinic_crystal()
        R = -o3.rand_matrix() if reflection else o3.rand_matrix()
        D_out = enc.irreps_out.D_from_matrix(R)

        y_ref = enc(Z, pos, mask, cell=cell)
        y_rot = enc(Z, pos @ R.T, mask, cell=cell @ R.T)
        torch.testing.assert_close(y_rot, y_ref @ D_out.T, atol=1e-9, rtol=1e-9)

        # Shifting individual atoms by integer lattice vectors must leave periodic features unchanged.
        shifts = torch.tensor([[[2.0, -1.0, 1.0], [-1.0, 0.0, 2.0], [0.0, 3.0, -2.0]]], dtype=torch.float64)
        pos_shifted = pos + shifts @ cell
        y_shifted = enc(Z, pos_shifted, mask, cell=cell)
        torch.testing.assert_close(y_shifted, y_ref, atol=1e-9, rtol=1e-9)
    finally:
        torch.set_default_dtype(previous)


def test_pbc_supercell_atom_feature_equivalence():
    torch.manual_seed(9)
    enc = Encoder(hidden_irreps="8x0e + 4x1o + 2x2e", n_layers=2, cutoff=6.5, n_elements=30).double().eval()
    Z, pos, mask, cell = make_triclinic_crystal()
    y_prim = enc(Z, pos, mask, cell=cell)

    # Construct a 2x1x1 supercell duplicating each basis atom at +a_1.
    a1 = cell[:, 0:1, :]
    Z_sup = torch.cat([Z, Z], dim=1)
    pos_sup = torch.cat([pos, pos + a1], dim=1)
    mask_sup = torch.ones_like(Z_sup, dtype=torch.bool)
    cell_sup = cell.clone()
    cell_sup[:, 0, :] *= 2.0

    y_sup = enc(Z_sup, pos_sup, mask_sup, cell=cell_sup)
    torch.testing.assert_close(y_sup[:, :3], y_prim, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(y_sup[:, 3:], y_prim, atol=1e-9, rtol=1e-9)


def test_pbc_self_image_neighbors_and_mixed_batch():
    torch.manual_seed(12)
    enc = Encoder(hidden_irreps="8x0e + 4x1o + 2x2e", n_layers=2, cutoff=6.5, n_elements=30).double().eval()
    # 1-atom cubic cell with a=4.5 bohr (< cutoff=6.5): isolated atom has no edges, periodic has self-images.
    Z1 = torch.tensor([[3]], dtype=torch.long)
    pos1 = torch.zeros(1, 1, 3, dtype=torch.float64)
    mask1 = torch.ones(1, 1, dtype=torch.bool)
    cell1 = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 4.5
    y_isolated = enc(Z1, pos1, mask1)
    y_crystal = enc(Z1, pos1, mask1, cell=cell1)
    assert not torch.allclose(y_isolated, y_crystal)

    # Mixed molecule + crystal batch in collate_geometry.
    mol = single_geometry([8, 1, 1], [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.0, 1.8, 0.0]])
    crys = single_geometry([3], [[0.0, 0.0, 0.0]])
    crys["cell"] = np.eye(3) * 4.5
    crys["pbc"] = np.array([True, True, True])
    collated = collate_geometry([mol, crys], dtype=torch.float64)
    assert collated["pbc"][0].tolist() == [False, False, False]
    assert collated["pbc"][1].tolist() == [True, True, True]
    y_batch = enc(collated["Z"], collated["pos"], collated["atom_mask"],
                  cell=collated["cell"], pbc=collated["pbc"])
    y_mol_only = enc(collated["Z"][:1], collated["pos"][:1], collated["atom_mask"][:1])
    torch.testing.assert_close(y_batch[:1], y_mol_only, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(y_batch[1:, :1], y_crystal, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("mode", ["prefix", "xattnA", "xattnB", "xattnC"])
def test_ylm_lm_pbc_o3_loss_invariance(mode):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        seed_everything(21)
        tok = CharTokenizer()
        cfg = YLMConfig(
            mode=mode, vocab_size=tok.vocab_size, bos_token_id=tok.bos_id, eos_token_id=tok.eos_id,
            n_layer=2, n_embd=16, n_head=2, n_positions=64, key_dim=8, cutoff=6.5,
            cross_irreps="2x0e + 2x1o + 1x1e + 1x2e",
        )
        model = YLMCausalLM(cfg).double().eval()
        for adapter in model.adapters:
            adapter.gate.data.fill_(0.5)

        Z, pos, _, cell = make_triclinic_crystal()
        geom = single_geometry(Z[0].numpy(), pos[0].numpy())
        geom["cell"] = cell[0].numpy()
        geom["pbc"] = np.array([True, True, True])

        R = -o3.rand_matrix().double()
        moved = dict(
            geom,
            pos=geom["pos"] @ R.numpy().T + np.array([2.0, -1.0, 0.5]),
            cell=geom["cell"] @ R.numpy().T,
        )
        ex1 = Example([Seg("charged crystal=LiFeO "), Seg("3.45", "answer")], geom)
        ex2 = Example([Seg("charged crystal=LiFeO "), Seg("3.45", "answer")], moved)
        b1 = collate([ex1], tok, 48, dtype=torch.float64)
        b2 = collate([ex2], tok, 48, dtype=torch.float64)
        o1 = model(**model_inputs(b1))
        o2 = model(**model_inputs(b2))
        torch.testing.assert_close(o1["loss"], o2["loss"], atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(o1["logits"], o2["logits"], atol=1e-7, rtol=1e-7)
    finally:
        torch.set_default_dtype(previous)


def test_mp_battery_pbc_loader_and_geometry_controls(tmp_path, monkeypatch):
    def make_fake_structure(saved):
        sites = [
            types.SimpleNamespace(
                is_ordered=True,
                specie=types.SimpleNamespace(Z=z),
                frac_coords=np.asarray(fc, dtype=np.float64),
            )
            for z, fc in zip(saved["zs"], saved["frac"])
        ]
        s = types.SimpleNamespace(
            lattice=types.SimpleNamespace(matrix=np.asarray(saved["lattice"], dtype=np.float64)),
            sites=sites,
        )
        # Support iteration and len() like pymatgen.core.Structure.
        class FakeStructure:
            def __init__(self, lattice, site_list):
                self.lattice = lattice
                self._sites = site_list

            def __iter__(self):
                return iter(self._sites)

            def __len__(self):
                return len(self._sites)

        return FakeStructure(s.lattice, sites)

    core = types.ModuleType("pymatgen.core")
    core.Structure = types.SimpleNamespace(from_dict=make_fake_structure)
    monkeypatch.setitem(sys.modules, "pymatgen", types.ModuleType("pymatgen"))
    monkeypatch.setitem(sys.modules, "pymatgen.core", core)
    monkeypatch.setenv("MP_API_KEY", "offline-test-only")

    path = tmp_path / "mp.json"
    path.write_text(json.dumps({
        "complete": True,
        "records": [
            dict(
                id="bat1", material_id="mp-1", formula="LiFeO2", working_ion="Li",
                structure={"zs": [3, 26, 8, 8], "frac": [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [1.75, -0.25, 0.75]],
                           "lattice": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]},
                targets={"average_voltage": 3.4},
            ),
            dict(
                id="bat2", material_id="mp-2", formula="LiFeO2", working_ion="Li",
                structure={"zs": [3, 26, 8, 8], "frac": [[0.1, 0, 0], [0.6, 0.5, 0.5], [0.3, 0.25, 0.25], [0.8, 0.75, 0.75]],
                           "lattice": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]},
                targets={"average_voltage": 3.6},
            ),
        ],
    }))
    rows = mp.load_mp_battery(cache_path=path, geom_mode="pbc")
    assert len(rows) == 2
    g0 = rows[0]["example"].geom
    assert g0["cell"].shape == (3, 3)
    np.testing.assert_allclose(g0["cell"], np.eye(3) * 4.0 * ANGSTROM_TO_BOHR)
    assert g0["pbc"].tolist() == [True, True, True]
    # Fractional coordinate 1.75 wraps into [0, 1) -> 0.75 * 4.0 Angstrom.
    np.testing.assert_allclose(g0["pos"][3], np.array([3.0, 3.0, 3.0]) * ANGSTROM_TO_BOHR)

    zeroed, rep_zero = geometry_control(rows, "zero", indices=[0])
    assert rep_zero["changed"] == 1
    assert np.all(zeroed[0]["example"].geom["cell"] == 0)
    assert not np.any(zeroed[0]["example"].geom["pbc"])

    donored, rep_donor = geometry_control(rows, "donor", indices=[0], donor_indices=[1], seed=0)
    assert rep_donor["changed"] == 1
    np.testing.assert_allclose(donored[0]["example"].geom["cell"], np.eye(3) * 5.0 * ANGSTROM_TO_BOHR)


def test_pyproject_toml_metadata():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    assert data["project"]["name"] == "ylm"
    assert data["tool"]["setuptools"]["packages"] == ["src", "src.tasks"]
    assert "crystals" in data["project"]["optional-dependencies"]

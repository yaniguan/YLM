"""Equivariance, parser, and convex-ceiling tests for ported v1 / quad / pol components."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from e3nn import o3
from rdkit import Chem

from src.baselines import PipelineComp
from src.model import YLM
from src.option_bc import CompModel
from src.parse_question import parse_pair, resolve_pol, resolve_program
from src.pol import PolModel, execute_pol, to_matrix
from src.prog import ProgModel
from src.quad import QuadModel, quad_to_2e


@pytest.fixture()
def float64_default():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _fake_batch(B=3, N=7, T=5, d=16):
    g = torch.Generator().manual_seed(0)
    Z = torch.randint(1, 10, (B, N), generator=g)
    pos = torch.randn((B, N, 3), generator=g, dtype=torch.float64) * 3.0
    atom_mask = torch.ones(B, N, dtype=torch.bool)
    atom_mask[0, -2:] = False
    text = torch.randn((B, T, d), generator=g, dtype=torch.float64)
    text_mask = torch.ones(B, T, dtype=torch.bool)
    task = torch.tensor([0, 1, 1])
    return Z, pos, atom_mask, text, text_mask, task


def test_pair_and_pol_parsers():
    a, b = "the carbon atom in a ring", "the oxygen atom bonded to two hydrogens"
    t, first, second = parse_pair(f"Subtract the force on {b} from the force on {a}.")
    assert t == "diff" and first["element"] == 6 and second["element"] == 8
    t, first, second = parse_pair(f"What is the projection of the force on {a} toward {b}?")
    assert t == "proj" and first["element"] == 6 and second["element"] == 8
    assert parse_pair(f"Is the force on {a} larger than on {b}?") is None

    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    task, ia, ib = resolve_program(
        "What is the force on the oxygen atom bonded to one carbon and one hydrogen?",
        mol,
    )
    assert task == "force" and ia >= 0 and ib == -1

    ptask, pa, pb = resolve_pol("What is the isotropic polarizability of the molecule?", mol)
    assert ptask == "pol_iso" and pa == -1 and pb == -1
    ptask2, pa2, pb2 = resolve_pol(
        "What dipole is induced when a unit electric field points from the carbon atom bonded to three hydrogens and one oxygen to the oxygen atom bonded to one carbon and one hydrogen?",
        mol,
    )
    assert ptask2 == "pol_ind" and pa2 >= 0 and pb2 >= 0 and pa2 != pb2


def test_pipeline_comp_improper_rotation_and_o3_parity(float64_default):
    Z, pos, mask, text, tmask, task = _fake_batch()
    pipe = PipelineComp()
    ptask = torch.tensor([0, 1, 2])
    atoms = torch.tensor([[0, -1], [1, 4], [2, 5]])
    y = pipe(Z, pos, mask, ptask, atoms)
    assert y[0].abs().max() > 0
    R = -o3.rand_matrix()
    assert torch.det(R) < 0
    assert (pipe(Z, pos @ R.T, mask, ptask, atoms) - y @ R.T).abs().max() < 1e-8

    for name, model in [("bc", CompModel(text_dim=16, parity="o3")), ("ylm", YLM(text_dim=16, parity="o3"))]:
        args = (Z, pos, mask, text, tmask) + ((task,) if name == "ylm" else ())
        args_r = (Z, pos @ R.T, mask, text, tmask) + ((task,) if name == "ylm" else ())
        out0, out_r = model(*args)[0], model(*args_r)[0]
        assert (out_r - out0 @ R.T).abs().max() < 1e-8, name


def test_prog_model_equivariant_and_permutation_invariant(float64_default):
    Z, pos, mask, text, tmask, _ = _fake_batch()
    for training in (True, False):
        model = ProgModel(text_dim=16)
        model.train(training)
        R = -o3.rand_matrix()
        y, _ = model(Z, pos, mask, text, tmask)
        y_r, _ = model(Z, pos @ R.T, mask, text, tmask)
        assert (y_r - y @ R.T).abs().max() < 1e-8
        perm = torch.tensor([2, 0, 1, 4, 3, 6, 5])
        y_p, _ = model(Z[:, perm], pos[:, perm], mask[:, perm], text, tmask)
        assert (y_p - y).abs().max() < 1e-8 or not training


def test_pol_model_tensor_equivariance_and_contractions(float64_default):
    Z, pos, mask, text, tmask, _ = _fake_batch()
    model = PolModel(text_dim=16, iso_mean=3.0, iso_std=2.0, s2=1.5)
    R = -o3.rand_matrix()
    D = o3.Irreps("2e").D_from_matrix(R)
    a0, a2, _, _ = model(Z, pos, mask, text, tmask)
    b0, b2, _, _ = model(Z, pos @ R.T, mask, text, tmask)
    assert (b0 - a0).abs().max() < 1e-8
    assert (b2 - a2 @ D.T).abs().max() < 1e-8

    A, B = to_matrix(a0, a2), to_matrix(b0, b2)
    assert (B - R @ A @ R.T).abs().max() < 1e-8

    ptask = torch.tensor([2, 3, 3])
    atoms = torch.tensor([[0, 3], [1, 4], [2, 5]])
    y = execute_pol(a0, a2, pos, ptask, atoms)
    yr = execute_pol(b0, b2, pos @ R.T, ptask, atoms)
    assert (yr[0, 0] - y[0, 0]).abs() < 1e-8
    assert (yr[1:, :3] - y[1:, :3] @ R.T).abs().max() < 1e-8
    zero_unresolved = execute_pol(a0, a2, pos, torch.tensor([2, -1, 3]), torch.tensor([[0, 3], [1, 4], [2, -1]]))
    assert zero_unresolved[1:].abs().max() == 0


def test_quadrupole_head_equivariant_l2(float64_default):
    Z, pos, mask, text, tmask, _ = _fake_batch()
    R = o3.rand_matrix()
    D = o3.Irreps("2e").D_from_matrix(R)
    A = torch.randn(3, 3)
    Q = A + A.T
    Q = Q - torch.eye(3) * torch.trace(Q) / 3.0
    assert (quad_to_2e(R @ Q @ R.T) - quad_to_2e(Q) @ D.T).abs().max() < 1e-9

    model = QuadModel(text_dim=16)
    y = model(Z, pos, mask, text, tmask)
    assert (model(Z, pos @ R.T, mask, text, tmask) - y @ D.T).abs().max() < 1e-8
    assert (model(Z, pos + 4.0, mask, text, tmask) - y).abs().max() < 1e-8


def test_convex_ceiling_projection():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("cc62", root / "scripts" / "62_convex_ceiling.py")
    cc62 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc62)

    F = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    y_inside = np.array([0.2, 0.3, 0.5])
    p = cc62.project_hull(F, y_inside)
    assert np.allclose(p, y_inside, atol=1e-4)

    y_outside = np.array([-1.0, -1.0, -1.0])
    p_hull = cc62.project_hull(F, y_outside)
    p_plus = cc62.project_hull_plus_sum(F, y_outside)
    assert np.linalg.norm(p_plus - y_outside) < np.linalg.norm(p_hull - y_outside)


def test_prog_and_pol_handle_empty_atom_mask(float64_default):
    Z, pos, mask, text, tmask, _ = _fake_batch()
    mask_empty = mask.clone()
    mask_empty[1] = False

    prog = ProgModel(text_dim=16)
    for training in (True, False):
        prog.train(training)
        y, aux = prog(Z, pos, mask_empty, text, tmask)
        assert torch.isfinite(y).all()
        assert torch.all(y[1] == 0)
        assert torch.isfinite(aux["task_logits"]).all()

    pol = PolModel(text_dim=16)
    a0, a2, tl, ptr = pol(Z, pos, mask_empty, text, tmask)
    assert torch.isfinite(a0).all()
    assert torch.isfinite(a2).all()
    assert torch.isfinite(tl).all()
    assert all(torch.isfinite(lg[1]).all() for lg in ptr)


def test_make_anonymous_scrubs_html_js_and_notebooks(tmp_path: Path):
    import subprocess
    import sys
    import zipfile

    root = Path(__file__).resolve().parents[1]
    out_zip = tmp_path / "anon.zip"
    res = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "63_make_anonymous.py"),
            "--repo-root",
            str(root),
            "--out",
            str(out_zip),
            "--extra-terms",
            "yaniguan",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "no forbidden strings found" in res.stdout
    with zipfile.ZipFile(out_zip) as zf:
        for check_path in ("site/index.html", "site/app.js", "colab/run_packs.ipynb"):
            content = zf.read(check_path).decode("utf-8")
            assert "yaniguan" not in content.lower()
            assert "anonymous" in content.lower()


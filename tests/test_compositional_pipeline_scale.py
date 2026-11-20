"""Tests for #87 Item 2 port: PipelineComp physical-unit scaling and T5 signed axis projection."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from e3nn import o3
import torch

from src.baselines import PipelineComp, execute

ROOT = Path(__file__).resolve().parents[1]


def _load_script(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_execute_signed_axis_projection_and_unresolved_zero():
    pos = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    # Force on atom 0 points along -x (-3, 4, 0), so signed projection onto u = +x must be (-3, 0, 0)
    v = torch.tensor([[[-3.0, 4.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]])
    ops = ("force", "diff", "proj")

    y_proj = execute(ops, v, pos, torch.tensor([2]), torch.tensor([[0, 1]]))
    assert torch.allclose(y_proj, torch.tensor([[-3.0, 0.0, 0.0]]), atol=1e-5)

    y_diff = execute(ops, v, pos, torch.tensor([1]), torch.tensor([[0, 1]]))
    assert torch.allclose(y_diff, torch.tensor([[-4.0, 4.0, 0.0]]), atol=1e-5)

    # Unresolved atom or task returns zero vector
    y_fail = execute(ops, v, pos, torch.tensor([2]), torch.tensor([[0, -1]]))
    assert torch.allclose(y_fail, torch.zeros(1, 3))


def test_pipeline_comp_equivariance_and_task_scale_normalization():
    torch.manual_seed(0)
    model = PipelineComp(hidden_irreps="8x0e + 4x1o + 2x2e", n_layers=1, ops=("diff", "proj"))
    model.eval()
    Z = torch.tensor([[8, 1, 1], [6, 8, 0]])
    pos = torch.randn(2, 3, 3)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    ptask = torch.tensor([0, 1])  # diff, proj
    atoms = torch.tensor([[0, 1], [0, 1]])

    R = o3.rand_matrix()
    y0 = model(Z, pos, mask, ptask, atoms)
    yR = model(Z, pos @ R.T, mask, ptask, atoms)
    assert torch.allclose(yR, y0 @ R.T, atol=1e-5)

    mod14 = _load_script("scripts/14_train_comp.py", "train_comp_14")
    scale = torch.tensor([0.092, 0.043])
    task = torch.tensor([0, 1])
    scaled_pred = mod14.forward(model, "pipeline", (Z, pos, mask, ptask, atoms), task, scale)
    recovered_phys = scaled_pred * scale[task][:, None]
    assert torch.allclose(recovered_phys, y0, atol=1e-6)


def test_t5_templates_use_signed_axis_projection_wording():
    mod13 = _load_script("scripts/13_make_compositional.py", "make_comp_13")
    proj_templates = mod13.TEMPLATES["proj"]
    assert "vector projection of the force on {a} onto the direction toward {b}" in proj_templates[0]
    assert not any("part of the force on {a} that points toward" in t for t in proj_templates)

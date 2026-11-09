"""CPU checks for the opt-in 0e+2e readout. The default vector head stays as it is."""
import importlib.util
import json
import math
import pickle
import sys
from pathlib import Path

import pytest
import torch
from e3nn import o3
from e3nn.io import CartesianTensor

from src.model import YLM
from src.tensor_readout import (
    TensorReadout,
    build_arrays,
    check_quadrupole_matrix,
    load_quadrupole_labels,
    make_batch,
    quadrupole_loss,
    rotate_matrix,
    split_rows,
    unique_entries,
)


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _small(option="A", tensor_readout=False, **kwargs):
    settings = dict(
        hidden_irreps="4x0e + 2x0o + 3x1o + 2x1e + 2x2e",
        cross_irreps="3x0e + 2x1o + 1x1e + 1x2e",
        n_layers=2,
        option=option,
        tensor_readout=tensor_readout,
    )
    settings.update(kwargs)
    return YLM(8, **settings)


def _inputs(batch=2):
    torch.manual_seed(0)
    Z = torch.randint(1, 9, (batch, 4))
    pos = torch.randn(batch, 4, 3)
    mask = torch.ones(batch, 4, dtype=torch.bool)
    mask[0, -1] = False
    text = torch.randn(batch, 3, 8)
    text_mask = torch.tensor([[True, True, False]] * batch)
    task = torch.zeros(batch, dtype=torch.long)
    return Z, pos, mask, text, text_mask, task


def _traceless(seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(3, 3, generator=gen, dtype=torch.float64)
    q = 0.5 * (q + q.T)
    return q - torch.eye(3, dtype=torch.float64) * q.trace() / 3


def test_head_matches_e3nn_and_uses_l2():
    head = TensorReadout("1x0e + 1x1o + 1x2e")
    features = torch.randn(4, head.linear.irreps_in.dim)
    irreps = head.linear(features)
    expected = CartesianTensor("ij=ji").to_cartesian(irreps)
    torch.testing.assert_close(head(features), expected, atol=1e-8, rtol=1e-8)
    assert "change_of_basis" not in head.state_dict()
    torch.testing.assert_close(head(features), head(features).transpose(-1, -2))

    only_l2 = torch.zeros(1, head.linear.irreps_in.dim)
    l2 = head.linear.irreps_in.slices()[-1]
    only_l2[..., l2] = 1
    matrix = head(only_l2)
    assert matrix.abs().sum() > 0
    assert matrix.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max() < 1e-8


def test_label_rotation_is_0e_plus_2e():
    q = _traceless(1)
    checked = check_quadrupole_matrix(q)
    cartesian = CartesianTensor("ij=ji")
    irreps = cartesian.from_cartesian(checked)
    assert irreps[0].abs() < 1e-8
    assert irreps[1:].abs().sum() > 0
    R = -o3.rand_matrix()
    got = cartesian.to_cartesian(irreps @ cartesian.D_from_matrix(R).T)
    torch.testing.assert_close(got, rotate_matrix(checked, R), atol=1e-8, rtol=1e-8)
    assert unique_entries(checked).shape == (6,)


def test_load_and_pack_quadrupole_labels(tmp_path):
    q = _traceless(2)
    path = tmp_path / "qm9_quad.pkl"
    with open(path, "wb") as handle:
        pickle.dump({3: q.numpy()}, handle)
    loaded = load_quadrupole_labels(path)
    geom = [{"numbers": [1], "pos": [[0.0, 0.0, 0.0]]}] * 4
    geom[3] = {"numbers": [8, 1], "pos": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}
    arrays = build_arrays(geom, loaded, n_max=4)
    assert arrays["mol"].tolist() == [3]
    assert arrays["mask"][0].tolist() == [True, True, False, False]
    torch.testing.assert_close(arrays["target"][0].double(), q, atol=1e-6, rtol=1e-6)
    rows = split_rows(torch.tensor([8, 9, 10]))
    assert rows["val"].tolist() == [0]
    assert rows["test"].tolist() == [1]
    assert rows["train"].tolist() == [2]
    with pytest.raises(ValueError):
        check_quadrupole_matrix(torch.eye(3))
    with pytest.raises(ValueError):
        check_quadrupole_matrix(torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))


@pytest.mark.parametrize("option", ["A", "B", "C"])
@pytest.mark.parametrize("reflection", [False, True])
def test_tensor_readout_equivariant(option, reflection):
    torch.manual_seed(4)
    model = _small(option, tensor_readout=True)
    assert model.cross.option == option
    Z, pos, mask, text, text_mask, task = _inputs()
    R = -o3.rand_matrix() if reflection else o3.rand_matrix()
    y, attn = model(Z, pos, mask, text, text_mask, task)
    yr, attn_r = model(Z, pos @ R.T, mask, text, text_mask, task)
    assert y.shape == (2, 3, 3)
    torch.testing.assert_close(y, y.transpose(-1, -2), atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(yr, rotate_matrix(y, R), atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(attn_r, attn, atol=2e-8, rtol=2e-8)
    y_move, attn_move = model(Z, pos + torch.tensor([2.0, -1.0, 4.0]), mask, text, text_mask, task)
    torch.testing.assert_close(y_move, y, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(attn_move, attn, atol=2e-8, rtol=2e-8)
    assert attn[0, -1].abs() == 0


def test_tensor_readout_permutation_and_gradient():
    model = _small("C", tensor_readout=True)
    Z, pos, mask, text, text_mask, task = _inputs(batch=1)
    y, attn = model(Z, pos, mask, text, text_mask, task)
    order = torch.tensor([2, 0, 3, 1])
    yp, attnp = model(Z[:, order], pos[:, order], mask[:, order], text, text_mask, task)
    torch.testing.assert_close(yp, y)
    torch.testing.assert_close(attnp, attn[:, order])
    pos = pos.detach().requires_grad_(True)
    pred, _ = model(Z, pos, mask, text, text_mask, task)
    pred.square().sum().backward()
    assert model.readout.linear.weight.grad.abs().sum() > 0
    assert pos.grad.abs().sum() > 0
    assert pos.grad[0, -1].abs().sum() == 0


def test_default_head_is_still_a_vector():
    model = _small()
    assert model.tensor_readout is False
    assert model.cross.option == "A"
    y, attn = model(*_inputs())
    assert y.shape == (2, 3)
    assert attn.shape == (2, 4)
    keys = set(model.state_dict())
    assert "readout.weight" in keys
    assert not any(key.startswith("readout.linear") for key in keys)
    with pytest.raises(ValueError):
        _small(tensor_readout=True, cross_irreps="2x1o")


def test_v0_checkpoint_still_loads():
    torch.set_default_dtype(torch.float32)
    path = Path(__file__).resolve().parents[1] / "checkpoints/v0/ylm_minilm_s0.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = YLM(text_dim=384, text_mode="text", option="A")
    assert set(model.state_dict()) == set(saved["state_dict"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    Z = torch.tensor([[8, 1, 1]])
    pos = torch.tensor([[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [-0.5, 1.3, 0.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    text = torch.zeros(1, 2, 384)
    text_mask = torch.ones(1, 2, dtype=torch.bool)
    with torch.no_grad():
        y, attn = model(Z, pos, mask, text, text_mask, torch.tensor([1]))
    assert y.shape == (1, 3) and attn.shape == (1, 3)
    assert torch.isfinite(y).all()
    other = YLM(text_dim=384, tensor_readout=True)
    with pytest.raises(RuntimeError):
        other.load_state_dict(saved["state_dict"])


def _train_script():
    path = Path(__file__).resolve().parents[1] / "scripts/16_train_quad.py"
    spec = importlib.util.spec_from_file_location("train_quad", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_hook_one_step_and_eval():
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    geom = [{"numbers": [1], "pos": [[0.0, 0.0, 0.0]]} for _ in range(10)]
    quad = {}
    for mol in (1, 8, 9):
        geom[mol] = {"numbers": [6, 1, 1], "pos": torch.randn(3, 3).tolist()}
        quad[mol] = _traceless(mol)
    arrays = build_arrays(geom, quad, n_max=4)
    rows = split_rows(arrays["mol"])
    assert rows["train"].tolist() == [0]
    assert rows["val"].tolist() == [1]
    assert rows["test"].tolist() == [2]
    model = _small("A", tensor_readout=True, text_mode="taskid", n_tasks=1)
    inputs, target = make_batch(arrays, rows["train"], "cpu")
    scale = unique_entries(target).pow(2).mean().sqrt()
    pred = model(*inputs)[0]
    loss = quadrupole_loss(pred, target / scale)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.readout.linear.weight.grad.abs().sum() > 0
    report = _train_script().evaluate(model, arrays, rows["test"], scale, "cpu", n_rot=2, batch_size=4)
    assert report["n"] == 1
    assert report["eq_err_mean"] < 1e-4
    assert math.isfinite(report["mae"])


def test_question_splits_and_script_smoke(tmp_path, monkeypatch):
    torch.set_default_dtype(torch.float32)
    module = _train_script()
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([
        {"mol": 1, "split": "train"},
        {"mol": 1, "split": "test_heldout"},
        {"mol": 8, "split": "test"},
        {"mol": 9, "split": "val"},
    ]))
    rows, dropped = module.splits_from_questions(torch.tensor([9, 1, 8]), questions)
    assert dropped == 0
    assert rows["train"].tolist() == [1]
    assert rows["val"].tolist() == [0]
    assert rows["test"].tolist() == [2]
    fallback, dropped = module.splits_from_questions(torch.tensor([9, 1, 8]), tmp_path / "missing.json")
    assert dropped == 0
    assert fallback["test"].tolist() == [0]
    assert fallback["train"].tolist() == [1]
    assert fallback["val"].tolist() == [2]

    geom = [{"numbers": [6, 1], "pos": [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]} for _ in range(10)]
    quad = {mol: _traceless(mol).numpy() for mol in range(10)}
    geom_path = tmp_path / "geom.pkl"
    quad_path = tmp_path / "quad.pkl"
    with open(geom_path, "wb") as handle:
        pickle.dump(geom, handle)
    with open(quad_path, "wb") as handle:
        pickle.dump(quad, handle)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "16_train_quad.py", "--name", "smoke", "--geom", str(geom_path), "--quad", str(quad_path),
        "--questions", str(tmp_path / "missing.json"), "--tiny", "--epochs", "1", "--max_steps", "1",
        "--eval_rotations", "2", "--batch_size", "4", "--device", "cpu",
    ])
    module.main()
    saved = json.loads((tmp_path / "runs" / "smoke" / "results.json").read_text())
    assert saved["args"]["option"] == "A"
    assert saved["test"]["n"] == 1
    assert saved["n_dropped"] == 0
    assert saved["test"]["eq_err_mean"] < 1e-4
    assert math.isfinite(saved["test"]["mae"])


def test_tensor_readout_is_the_last_argument():
    import inspect
    params = inspect.signature(YLM.__init__).parameters
    names = list(params)
    assert names[-1] == "tensor_readout"
    assert names[-2] == "b_score_mode"
    assert params["tensor_readout"].default is False
    assert params["b_score_mode"].default == "legacy"
    assert params["option"].default == "A"


def test_quad_molecule_missing_from_questions(tmp_path, monkeypatch):
    """One quadrupole molecule is absent from the question file. It must be dropped."""
    torch.set_default_dtype(torch.float32)
    module = _train_script()
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([
        {"mol": 1, "split": "train"},
        {"mol": 8, "split": "val"},
        {"mol": 9, "split": "test"},
        {"mol": 9, "split": "test_heldout"},
    ]))
    rows, dropped = module.splits_from_questions(torch.tensor([1, 4, 8, 9]), questions)
    assert dropped == 1
    assert rows["train"].tolist() == [0]
    assert rows["val"].tolist() == [2]
    assert rows["test"].tolist() == [3]
    for split_rows in rows.values():
        assert 1 not in split_rows.tolist()
    geom = [{"numbers": [6, 1], "pos": [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]} for _ in range(10)]
    # Molecule 4 is in the quadrupole file and not in the question file.
    quad = {mol: _traceless(mol).numpy() for mol in (1, 4, 8, 9)}
    geom_path = tmp_path / "geom.pkl"
    quad_path = tmp_path / "quad.pkl"
    with open(geom_path, "wb") as handle:
        pickle.dump(geom, handle)
    with open(quad_path, "wb") as handle:
        pickle.dump(quad, handle)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "16_train_quad.py", "--name", "missing_split", "--geom", str(geom_path),
        "--quad", str(quad_path), "--questions", str(questions), "--tiny",
        "--epochs", "1", "--max_steps", "1", "--eval_rotations", "2",
        "--batch_size", "4", "--device", "cpu",
    ])
    module.main()
    log = (tmp_path / "runs" / "missing_split" / "log.txt").read_text()
    assert "molecules dropped (no question split): 1" in log
    saved = json.loads((tmp_path / "runs" / "missing_split" / "results.json").read_text())
    assert saved["n_dropped"] == 1
    assert saved["test"]["n"] == 1
    assert saved["val"]["n"] == 1

"""CPU checks for the opt-in 0e+2e readout. The default vector head stays as it is."""
import importlib.util
import json
import math
import pickle
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import src  # noqa: F401,E402  # allowlist e3nn 0.4.4 constants before importing e3nn

from e3nn import o3
from e3nn.io import CartesianTensor

from src.model import YLM
from src.tensor_readout import (
    TensorReadout,
    build_arrays,
    check_quadrupole_matrix,
    load_quadrupole_labels,
    make_batch,
    quadrupole_error,
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


def test_quadrupole_loss_and_error_are_o3_invariant():
    torch.manual_seed(17)
    raw_p = torch.randn(6, 3, 3, dtype=torch.float64)
    raw_t = torch.randn(6, 3, 3, dtype=torch.float64)
    pred = 0.5 * (raw_p + raw_p.transpose(-1, -2))
    target = 0.5 * (raw_t + raw_t.transpose(-1, -2))

    base_loss = quadrupole_loss(pred, target)
    base_err = quadrupole_error(pred, target)
    for sign in (1.0, -1.0):
        R = sign * o3.rand_matrix()
        rot_p = rotate_matrix(pred, R)
        rot_t = rotate_matrix(target, R)
        torch.testing.assert_close(quadrupole_loss(rot_p, rot_t), base_loss, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(quadrupole_error(rot_p, rot_t), base_err, atol=1e-14, rtol=1e-14)

    # Off-diagonal pure shear rotated by 45 degrees into a diagonal matrix must keep the exact same loss (2/9)
    off_diag = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float64)
    s = math.sqrt(0.5)
    r45 = torch.tensor([[s, s, 0.0], [-s, s, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    diag_rot = rotate_matrix(off_diag, r45)
    zeros = torch.zeros_like(off_diag)
    torch.testing.assert_close(quadrupole_loss(off_diag, zeros), torch.tensor(2.0 / 9.0, dtype=torch.float64))
    torch.testing.assert_close(quadrupole_loss(diag_rot, zeros), torch.tensor(2.0 / 9.0, dtype=torch.float64))


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


_V0_REFERENCE_PATH = Path(__file__).resolve().parent / "fixtures" / "v0_reference.json"
_V0_CHECKPOINTS = ("ylm_minilm_s0.pt", "ylm_matscibert_s0.pt", "baseline_taskid_s0.pt")
_V0_TEXT_DIMS = {"minilm": 384, "matscibert": 768, "taskid": 16}


def _v0_reference_inputs(text_dim: int):
    """Fixed non-coplanar 4-atom molecule (+ 1 padded slot) and deterministic text tensor."""
    Z = torch.tensor([[8, 6, 7, 1, 0]], dtype=torch.long)
    pos = torch.tensor(
        [[[0.0, 0.0, 0.2], [2.4, 0.1, -0.3], [-0.6, 1.8, 0.5], [1.1, -1.5, 1.4], [0.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    mask = torch.tensor([[True, True, True, True, False]])
    idx = torch.arange(3 * text_dim, dtype=torch.float64).reshape(1, 3, text_dim)
    text = torch.sin(idx * 0.37) + 0.5 * torch.cos(idx * 0.19)
    text_mask = torch.tensor([[True, True, False]])
    return Z, pos, mask, text, text_mask


def _eval_v0_checkpoint(ckpt_name: str, **ylm_overrides) -> dict:
    ckpt_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "v0"
    saved = torch.load(ckpt_dir / ckpt_name, map_location="cpu", weights_only=False)
    text_kind = saved["text"]
    text_dim = _V0_TEXT_DIMS[text_kind]
    kwargs = dict(text_dim=text_dim, text_mode=("taskid" if text_kind == "taskid" else "text"))
    kwargs.update(ylm_overrides)
    model = YLM(**kwargs)
    assert set(model.state_dict()) == set(saved["state_dict"])
    model.load_state_dict(saved["state_dict"])
    model = model.double().eval()
    Z, pos, mask, text, text_mask = _v0_reference_inputs(text_dim)
    with torch.no_grad():
        y_dip, attn_dip = model(Z, pos, mask, text, text_mask, torch.tensor([0]))
        y_frc, attn_frc = model(Z, pos, mask, text, text_mask, torch.tensor([1]))
    return {
        "text": text_kind,
        "text_dim": text_dim,
        "dipole_vec": y_dip[0].tolist(),
        "dipole_attn": attn_dip[0].tolist(),
        "force_vec": y_frc[0].tolist(),
        "force_attn": attn_frc[0].tolist(),
    }


def write_v0_reference(path: Path = _V0_REFERENCE_PATH) -> None:
    """Regenerate `tests/fixtures/v0_reference.json` from `checkpoints/v0/*.pt`.

    Run from the repo root:
      python tests/test_tensor_readout.py --regen-v0-reference
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        payload = {
            "note": "Reference outputs for checkpoints/v0/*.pt; regenerate with `python tests/test_tensor_readout.py --regen-v0-reference`.",
            "checkpoints": {name: _eval_v0_checkpoint(name) for name in _V0_CHECKPOINTS},
        }
    finally:
        torch.set_default_dtype(previous)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_v0_checkpoint_still_loads():
    # Regenerate reference file if intentional: `python tests/test_tensor_readout.py --regen-v0-reference`
    torch.set_default_dtype(torch.float32)
    reference = json.loads(_V0_REFERENCE_PATH.read_text(encoding="utf-8"))["checkpoints"]
    assert set(reference) == set(_V0_CHECKPOINTS)

    for name in _V0_CHECKPOINTS:
        got = _eval_v0_checkpoint(name)
        ref = reference[name]
        for key in ("dipole_vec", "dipole_attn", "force_vec", "force_attn"):
            torch.testing.assert_close(
                torch.tensor(got[key], dtype=torch.float64),
                torch.tensor(ref[key], dtype=torch.float64),
                atol=1e-6,
                rtol=1e-6,
            )

    # Changing `YLM(...)` defaults must fail either `load_state_dict` or the numerical check
    ckpt_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "v0"
    saved = torch.load(ckpt_dir / "ylm_minilm_s0.pt", map_location="cpu", weights_only=False)
    for bad_kwargs in (
        dict(tensor_readout=True),
        dict(option="B"),
        dict(option="C"),
        dict(n_cross_layers=2),
        dict(n_layers=2),
    ):
        other = YLM(text_dim=384, **bad_kwargs)
        with pytest.raises(RuntimeError):
            other.load_state_dict(saved["state_dict"])

    # Shape-preserving default changes still fail the numerical reference check (`atol=1e-6`)
    wrong_mode = _eval_v0_checkpoint("ylm_minilm_s0.pt", text_mode="taskid")
    mode_diff = (
        torch.tensor(wrong_mode["dipole_vec"], dtype=torch.float64)
        - torch.tensor(reference["ylm_minilm_s0.pt"]["dipole_vec"], dtype=torch.float64)
    ).abs().max().item()
    assert mode_diff > 1e-4

    wrong_parity = _eval_v0_checkpoint("baseline_taskid_s0.pt", legacy_odd_scalars=True)
    parity_diff = (
        torch.tensor(wrong_parity["dipole_vec"], dtype=torch.float64)
        - torch.tensor(reference["baseline_taskid_s0.pt"]["dipole_vec"], dtype=torch.float64)
    ).abs().max().item()
    assert parity_diff > 1e-4


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
    scale = target.pow(2).mean().sqrt()
    pred = model(*inputs)[0]
    loss = quadrupole_loss(pred, target / scale)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.readout.linear.weight.grad.abs().sum() > 0
    rep1 = _train_script().evaluate(model, arrays, rows["test"], scale, "cpu", n_rot=1, batch_size=4)
    report = _train_script().evaluate(model, arrays, rows["test"], scale, "cpu", n_rot=4, batch_size=4)
    assert report["n"] == 1
    assert report["eq_err_mean"] < 1e-4
    assert math.isfinite(report["mae"])
    assert abs(report["mae"] - rep1["mae"]) < 1e-5


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
    assert names[-2] == "n_cross_layers"
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


if __name__ == "__main__":
    if "--regen-v0-reference" in sys.argv[1:]:
        write_v0_reference()
        print(f"Wrote {_V0_REFERENCE_PATH}")
    else:
        raise SystemExit("Usage: python tests/test_tensor_readout.py --regen-v0-reference")

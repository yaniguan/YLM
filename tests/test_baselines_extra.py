"""Tests for EquiLLMBaseline and FrameAveragingBaseline."""
import pytest
import torch
from e3nn import o3

from src.baselines import EquiLLMBaseline, FrameAveragingBaseline


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def asymmetric_batch():
    torch.manual_seed(4)
    Z = torch.tensor([[8, 6, 1, 7, 0], [6, 8, 1, 1, 6]])
    # Asymmetric 3D positions so covariance eigenvalues are well-separated.
    pos = torch.tensor([
        [[0.0, 0.0, 0.0], [2.3, 0.2, 0.0], [-0.8, 1.9, 0.1], [0.4, -0.7, 3.1], [9.0, 9.0, 9.0]],
        [[0.1, -0.2, 0.0], [2.6, 0.3, -0.1], [-1.1, 1.7, 0.2], [0.2, -1.3, 2.9], [-2.0, -0.4, -1.8]],
    ])
    atom_mask = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]])
    text = torch.randn(2, 4, 12)
    text_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    task = torch.tensor([0, 1])
    return Z, pos, atom_mask, text, text_mask, task


@pytest.mark.parametrize("reflection", [False, True])
@pytest.mark.parametrize("text_mode", ["text", "taskid"])
@pytest.mark.parametrize("kind", ["equillm", "frame_avg"])
def test_baselines_o3_equivariance_translation_permutation_and_padding(kind, text_mode, reflection):
    Z, pos, atom_mask, text, text_mask, task = asymmetric_batch()
    if kind == "equillm":
        model = EquiLLMBaseline(12, hidden_irreps="4x0e + 2x0o + 3x1o + 2x1e + 1x2e",
                                n_layers=2, d=16, n_vec=4, n_bridge_layers=1, text_mode=text_mode).eval()
    else:
        model = FrameAveragingBaseline(12, d=16, n_atom_layers=1, text_mode=text_mode).eval()

    R = -o3.rand_matrix() if reflection else o3.rand_matrix()
    shift = torch.tensor([4.5, -3.0, 2.25])

    y, attn = model(Z, pos, atom_mask, text, text_mask, task)
    y_rot, attn_rot = model(Z, pos @ R.T, atom_mask, text, text_mask, task)
    y_shift, attn_shift = model(Z, pos + shift, atom_mask, text, text_mask, task)

    torch.testing.assert_close(y_rot, y @ R.T, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(attn_rot, attn, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(y_shift, y, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(attn_shift, attn, atol=1e-7, rtol=1e-7)
    assert attn[0, -1].item() == 0.0
    torch.testing.assert_close(attn.sum(-1), torch.ones(2), atol=1e-7, rtol=1e-7)

    perm = torch.tensor([3, 1, 4, 0, 2])
    y_perm, attn_perm = model(Z[:, perm], pos[:, perm], atom_mask[:, perm], text, text_mask, task)
    torch.testing.assert_close(y_perm, y, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(attn_perm, attn[:, perm], atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("kind", ["equillm", "frame_avg"])
def test_baselines_planar_water_and_gradients(kind):
    # Planar H2O has eigenvalues (0, lambda_2, lambda_3) which remain distinct.
    Z = torch.tensor([[8, 1, 1]])
    pos = torch.tensor([[[0.0, 0.0, 0.0], [1.81, 0.0, 0.0], [-0.45, 1.75, 0.0]]], requires_grad=True)
    atom_mask = torch.ones(1, 3, dtype=torch.bool)
    text = torch.randn(1, 3, 8, requires_grad=True)
    text_mask = torch.ones(1, 3, dtype=torch.bool)
    task = torch.tensor([1])
    model = (EquiLLMBaseline(8, hidden_irreps="4x0e + 2x1o", n_layers=1, d=16, n_vec=4, n_bridge_layers=1)
             if kind == "equillm" else FrameAveragingBaseline(8, d=16, n_atom_layers=1))
    R = -o3.rand_matrix()
    y, _ = model(Z, pos, atom_mask, text, text_mask, task)
    yr, _ = model(Z, pos @ R.T, atom_mask, text, text_mask, task)
    torch.testing.assert_close(yr, y @ R.T, atol=1e-7, rtol=1e-7)
    y.square().sum().backward()
    assert pos.grad is not None and torch.isfinite(pos.grad).all() and pos.grad.abs().sum() > 0
    assert text.grad is not None and torch.isfinite(text.grad).all() and text.grad.abs().sum() > 0

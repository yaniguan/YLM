"""Tests for multi-layer two-stream interactive YLM (--n_cross_layers)."""
import pytest
import torch
from e3nn import o3

from src.model import YLM


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def test_single_layer_default_preserves_legacy_state_dict():
    torch.manual_seed(10)
    a = YLM(16, hidden_irreps="4x0e + 2x1o", cross_irreps="2x0e + 2x1o", n_layers=1)
    torch.manual_seed(10)
    b = YLM(16, hidden_irreps="4x0e + 2x1o", cross_irreps="2x0e + 2x1o", n_layers=1, n_cross_layers=1)
    assert not hasattr(a, "extra_steps")
    assert a.state_dict().keys() == b.state_dict().keys()
    for k, v in a.state_dict().items():
        torch.testing.assert_close(v, b.state_dict()[k])


def test_invalid_n_cross_layers_rejected():
    with pytest.raises(ValueError, match="n_cross_layers"):
        YLM(16, n_cross_layers=0)


@pytest.mark.parametrize("option", ["A", "B", "C"])
@pytest.mark.parametrize("reflection", [False, True])
def test_multilayer_ylm_o3_padding_permutation_and_grad(option, reflection):
    torch.manual_seed(21)
    model = YLM(8, hidden_irreps="4x0e + 2x0o + 3x1o + 2x1e + 2x2e",
                cross_irreps="3x0e + 2x1o + 1x1e + 1x2e",
                n_layers=2, option=option, n_cross_layers=3)
    Z = torch.tensor([[6, 1, 8, 0]])
    pos = torch.randn(1, 4, 3)
    mask = torch.tensor([[True, True, True, False]])
    text = torch.randn(1, 3, 8)
    tm = torch.tensor([[True, True, False]])
    task = torch.tensor([1])

    R = -o3.rand_matrix() if reflection else o3.rand_matrix()
    y, a = model(Z, pos, mask, text, tm, task)
    yr, ar = model(Z, pos @ R.T, mask, text, tm, task)
    ym, am = model(Z, pos + torch.tensor([3.0, -1.5, 2.0]), mask, text, tm, task)
    torch.testing.assert_close(yr, y @ R.T, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(ar, a, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(ym, y, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(am, a, atol=2e-8, rtol=2e-8)

    # Atom permutation equivariance/invariance.
    perm = torch.tensor([3, 1, 0, 2])
    yp, ap = model(Z[:, perm], pos[:, perm], mask[:, perm], text, tm, task)
    torch.testing.assert_close(yp, y, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(ap, a[:, perm], atol=2e-8, rtol=2e-8)

    # Masked garbage text tokens must not affect outputs.
    text2 = torch.cat([text[:, :2], torch.randn(1, 4, 8) * 100.0], dim=1)
    tm2 = torch.tensor([[True, True, False, False, False, False]])
    yp2, ap2 = model(Z, pos, mask, text2, tm2, task)
    torch.testing.assert_close(yp2, y, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(ap2, a, atol=2e-8, rtol=2e-8)

    # Gradients flow through both initial cross layer and feedback blocks; padded slots stay zero.
    pos.requires_grad_(True)
    text.requires_grad_(True)
    model(Z, pos, mask, text, tm, task)[0].square().sum().backward()
    for grad in [pos.grad, text.grad, model.cross.q.weight.grad,
                 model.extra_steps[0].inv_mlp[0].weight.grad,
                 model.extra_steps[-1].cross.q.weight.grad]:
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert text.grad[:, 2].abs().sum() == 0
    assert pos.grad[:, 3].abs().sum() == 0

import pytest
import torch
from e3nn import o3
from src.irreps_to_text import IrrepsToText
from src.model import YLM


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture
def inputs():
    torch.manual_seed(42)
    return (torch.tensor([[6, 1, 8, 0]]), torch.randn(1, 4, 3),
            torch.tensor([[True, True, True, False]]), torch.randn(1, 3, 8),
            torch.tensor([[True, True, False]]), torch.tensor([0]))


@pytest.fixture
def model():
    return YLM(8, hidden_irreps='4x0e + 2x0o + 3x1o + 2x1e + 2x2e',
               cross_irreps='3x0e + 2x1o + 1x1e + 1x2e', n_layers=2)


@pytest.mark.parametrize('reflection', [False, True])
def test_cross_o3(reflection):
    torch.manual_seed(12)
    irreps = o3.Irreps('2x0e + 2x0o + 2x1o + 2x1e + 1x2e')
    layer = IrrepsToText(8, irreps, irreps)
    x, h = torch.randn(2, 5, irreps.dim), torch.randn(2, 3, 8)
    mask = torch.tensor([[True]*4+[False], [True]*5])
    R = -o3.rand_matrix() if reflection else o3.rand_matrix()
    D = irreps.D_from_matrix(R)
    y, a = layer(h, x, mask)
    yr, ar = layer(h, x @ D.T, mask)
    torch.testing.assert_close(yr, y @ D.T, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(ar, a, atol=2e-8, rtol=2e-8)
    perm = torch.tensor([4, 2, 0, 3, 1])
    yp, ap = layer(h, x[:, perm], mask[:, perm])
    torch.testing.assert_close(yp, y)
    torch.testing.assert_close(ap, a[:, :, perm])
    # Token operations are independent: adding tokens preserves existing outputs.
    padded, _ = layer(torch.cat([h, torch.randn(2, 2, 8)], 1), x, mask)
    torch.testing.assert_close(padded[:, :3], y)
    x.requires_grad_(); h.requires_grad_()
    layer(h, x, mask)[0].square().sum().backward()
    for grad in [x.grad, h.grad, layer.q.weight.grad, layer.v.weight.grad]:
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize('reflection', [False, True])
def test_ylm_o3(model, inputs, reflection):
    Z, pos, mask, text, tm, task = inputs
    R = -o3.rand_matrix() if reflection else o3.rand_matrix()
    y, a = model(*inputs)
    yr, ar = model(Z, pos @ R.T, mask, text, tm, task)
    torch.testing.assert_close(yr, y @ R.T, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(ar, a, atol=2e-8, rtol=2e-8)


def test_ylm_padding_permutation_grad(model, inputs):
    Z, pos, mask, text, tm, task = inputs
    y, a = model(*inputs)
    p = torch.tensor([3, 1, 0, 2])
    yp, ap = model(Z[:, p], pos[:, p], mask[:, p], text, tm, task)
    torch.testing.assert_close(yp, y)
    torch.testing.assert_close(ap, a[:, p])
    text2 = torch.cat([text[:, :2], torch.randn(1, 4, 8)*100], 1)
    tm2 = torch.tensor([[True, True, False, False, False, False]])
    yp, ap = model(Z, pos, mask, text2, tm2, task)
    torch.testing.assert_close(yp, y)
    torch.testing.assert_close(ap, a)
    pos.requires_grad_(); text.requires_grad_()
    model(Z, pos, mask, text, tm, task)[0].square().sum().backward()
    for grad in [pos.grad, text.grad, model.encoder.embed.weight.grad,
                 model.cross.q.weight.grad, model.readout.weight.grad]:
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert text.grad[:, 2].abs().sum() == 0
    assert pos.grad[:, 3].abs().sum() == 0

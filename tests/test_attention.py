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


@pytest.fixture(params=["A", "B", "C"])
def model(request):
    return YLM(8, hidden_irreps='4x0e + 2x0o + 3x1o + 2x1e + 2x2e',
               cross_irreps='3x0e + 2x1o + 1x1e + 1x2e', n_layers=2, option=request.param)


@pytest.mark.parametrize('reflection', [False, True])
@pytest.mark.parametrize("option", ["A", "B", "C"])
def test_cross_o3(reflection, option):
    torch.manual_seed(12)
    irreps = o3.Irreps('2x0e + 2x0o + 2x1o + 2x1e + 1x2e')
    layer = IrrepsToText(8, irreps, irreps, option=option)
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
    ym, am = model(Z, pos + torch.tensor([2., -1., 4.]), mask, text, tm, task)
    torch.testing.assert_close(ym, y, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(am, a, atol=2e-8, rtol=2e-8)


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


def test_c_creates_quadrupole_and_text_weights_receive_gradients():
    # Start with vectors only: A cannot synthesize l=2, whereas C's 1o x 1o
    # path can. A mixed side stream supplies vectors to the self product.
    layer = IrrepsToText(8, '2x1o', '2x1o + 1x2e', option='C')
    x = torch.randn(1, 4, 6)
    h = torch.randn(1, 2, 8)
    mask = torch.ones(1, 4, dtype=torch.bool)
    y, _ = layer(h, x, mask)
    assert y[..., -5:].abs().sum() > 0
    y[..., -5:].square().sum().backward()
    assert layer.tp_weights[-1].weight.grad.abs().sum() > 0
    R = -o3.rand_matrix()
    yr, _ = layer(h, x @ o3.Irreps('2x1o').D_from_matrix(R).T, mask)
    torch.testing.assert_close(yr, y @ layer.out_irreps.D_from_matrix(R).T, atol=2e-8, rtol=2e-8)


def test_b_direction_scores_change_for_relative_alignment():
    layer = IrrepsToText(8, '1x1o', '1x1o', option='B')
    with torch.no_grad():
        layer.q.weight.zero_(); layer.q.bias.zero_()
        layer.direction_key.weight.fill_(1)
        layer.direction_query.weight.zero_(); layer.direction_query.bias.fill_(1)
    h = torch.zeros(1, 1, 8)
    mask = torch.ones(1, 3, dtype=torch.bool)
    # All norms stay one, but global/atom alignment changes.
    x = torch.tensor([[[1., 0, 0], [1., 0, 0], [0, 1., 0]]])
    x2 = x.clone(); x2[0, 1, 0] = -1
    a = layer(h, x, mask)[1]
    a2 = layer(h, x2, mask)[1]
    assert (a-a2).abs().max() > 0.01

@pytest.mark.parametrize('reflection', [False, True])
def test_normalized_b(reflection):
    irreps = o3.Irreps('2x0e + 2x1o + 1x2e')
    layer = IrrepsToText(8, irreps, irreps, option='B', b_score_mode='normalized')
    h, x = torch.randn(2, 3, 8), torch.randn(2, 4, irreps.dim)
    mask = torch.tensor([[True, True, True, False]] * 2)
    a = IrrepsToText(8, irreps, irreps)
    a.load_state_dict(layer.state_dict(), strict=False)
    torch.testing.assert_close(layer(h, x, mask)[1], a(h, x, mask)[1])
    layer.direction_coefficient.data.fill_(0.7)
    D = irreps.D_from_matrix(o3.rand_matrix() * (-1 if reflection else 1))
    y, attn = layer(h, x, mask)
    yr, ar = layer(h, x @ D.T, mask)
    torch.testing.assert_close(ar, attn, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(yr, y @ D.T, atol=1e-7, rtol=1e-7)
    perm = torch.tensor([3, 1, 0, 2])
    torch.testing.assert_close(layer(h, x[:, perm], mask[:, perm])[1], attn[:, :, perm])
    zero = torch.zeros_like(x, requires_grad=True)
    layer(h, zero, mask)[0].sum().backward()
    assert torch.isfinite(zero.grad).all()
    assert torch.isfinite(layer.direction_coefficient.grad)


def test_b_legacy_default():
    torch.manual_seed(9)
    default = IrrepsToText(8, '2x0e + 1x1o', '1x1o', option='B')
    torch.manual_seed(9)
    explicit = IrrepsToText(8, '2x0e + 1x1o', '1x1o', option='B', b_score_mode='legacy')
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for k, v in default.state_dict().items():
        torch.testing.assert_close(v, explicit.state_dict()[k])

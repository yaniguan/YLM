"""Tests for Option C channel-wise ('uvu') tensor-product mode."""
import pytest
import torch
from e3nn import o3

from src.irreps_to_text import IrrepsToText
from src.model import YLM
from src.ylm_lm import YLMConfig, YLMCausalLM
from src.tasks.common import Example, Seg, collate, model_inputs, seed_everything
from src.tasks.tokenizer import CharTokenizer
from src.tasks.conformers import single_geometry


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def test_c_default_preserves_legacy_state_dict():
    torch.manual_seed(7)
    default = IrrepsToText(8, "2x0e + 2x1o", "2x1o + 1x2e", option="C")
    torch.manual_seed(7)
    explicit = IrrepsToText(8, "2x0e + 2x1o", "2x1o + 1x2e", option="C", c_tp_mode="fully_connected")
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key])


def test_invalid_c_tp_mode_rejected():
    with pytest.raises(ValueError, match="Unknown C tensor-product mode"):
        IrrepsToText(8, "2x1o", "2x1o", option="C", c_tp_mode="invalid")


@pytest.mark.parametrize("reflection", [False, True])
def test_channelwise_c_o3_equivariance_permutation_and_grad(reflection):
    torch.manual_seed(19)
    irreps = o3.Irreps("2x0e + 2x0o + 2x1o + 2x1e + 1x2e")
    layer = IrrepsToText(8, irreps, irreps, option="C", c_tp_mode="channelwise")
    x = torch.randn(2, 5, irreps.dim)
    h = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, True, True, True, False], [True] * 5])
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

    x.requires_grad_(True)
    h.requires_grad_(True)
    layer(h, x, mask)[0].square().sum().backward()
    for grad in [x.grad, h.grad, layer.tp_weights[-1].weight.grad, layer.tp_after.weight.grad]:
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_channelwise_c_synthesizes_quadrupole_and_pseudovector():
    torch.manual_seed(11)
    layer = IrrepsToText(8, "2x1o", "2x1o + 1x1e + 1x2e", option="C", c_tp_mode="channelwise")
    x = torch.randn(1, 4, 6)
    h = torch.randn(1, 2, 8)
    mask = torch.ones(1, 4, dtype=torch.bool)
    y, _ = layer(h, x, mask)
    # Both 1e (cross product) and 2e (symmetric traceless tensor) are formed from 1o inputs.
    assert y[..., 6:9].abs().sum() > 0
    assert y[..., 9:14].abs().sum() > 0
    y[..., 6:].square().sum().backward()
    assert layer.tp_weights[-1].weight.grad.abs().sum() > 0


def test_channelwise_c_reduces_default_parameter_count():
    fc_params = sum(p.numel() for p in YLM(32, option="C", c_tp_mode="fully_connected").parameters())
    cw_params = sum(p.numel() for p in YLM(32, option="C", c_tp_mode="channelwise").parameters())
    assert cw_params < fc_params / 10


def test_channelwise_c_ylm_and_causal_lm_invariance():
    torch.manual_seed(5)
    model = YLM(8, hidden_irreps="4x0e + 2x0o + 3x1o + 2x1e + 2x2e",
                cross_irreps="3x0e + 2x1o + 1x1e + 1x2e", n_layers=2,
                option="C", c_tp_mode="channelwise")
    Z = torch.tensor([[6, 1, 8, 0]])
    pos = torch.randn(1, 4, 3)
    mask = torch.tensor([[True, True, True, False]])
    text = torch.randn(1, 3, 8)
    tm = torch.tensor([[True, True, False]])
    task = torch.tensor([1])
    R = -o3.rand_matrix()
    y, a = model(Z, pos, mask, text, tm, task)
    yr, ar = model(Z, pos @ R.T, mask, text, tm, task)
    torch.testing.assert_close(yr, y @ R.T, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(ar, a, atol=2e-8, rtol=2e-8)

    torch.set_default_dtype(torch.float32)
    seed_everything(3)
    tok = CharTokenizer()
    cfg = YLMConfig(mode="xattnC", c_tp_mode="channelwise", vocab_size=tok.vocab_size,
                    n_layer=1, n_embd=16, n_head=2, n_positions=96, key_dim=8,
                    cross_irreps="2x0e + 2x1o + 1x1e + 1x2e")
    lm = YLMCausalLM(cfg).eval()
    for adapter in lm.adapters:
        adapter.gate.data.fill_(0.7)
    geom = single_geometry([8, 1, 1], [[0, 0, 0], [1.4, 0.2, 0], [-0.4, 1.3, 0.1]])
    rot = o3.rand_matrix().float().numpy()
    moved = dict(geom, pos=geom["pos"] @ rot.T + [3.0, -2.0, 1.0])
    ba = collate([Example([Seg("water E="), Seg("-1.2", "num")], geom)], tok, 40)
    bb = collate([Example([Seg("water E="), Seg("-1.2", "num")], moved)], tok, 40)
    with torch.no_grad():
        oa, ob = lm(**model_inputs(ba)), lm(**model_inputs(bb))
    torch.testing.assert_close(oa["loss"], ob["loss"], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(oa["logits"], ob["logits"], atol=3e-5, rtol=3e-5)

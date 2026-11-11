"""Tests for <vec> O(3)-equivariant readout on YLMCausalLM."""
import numpy as np
import pytest
import torch
from e3nn import o3

from src.ylm_lm import YLMConfig, YLMCausalLM
from src.tasks.common import Example, Seg, collate, decode_with_vectors, model_inputs, seed_everything
from src.tasks.conformers import single_geometry
from src.tasks.tokenizer import CharTokenizer, VEC_TOKEN


def water_geom():
    return single_geometry([8, 1, 1], [[0.0, 0.0, 0.0], [1.4, 0.2, 0.0], [-0.4, 1.3, 0.1]])


def make_vec_cfg(mode="xattnA", tok=None, **kw):
    tok = tok or CharTokenizer()
    return YLMConfig(
        mode=mode, vocab_size=tok.vocab_size, bos_token_id=tok.bos_id, eos_token_id=tok.eos_id,
        n_layer=2, n_embd=16, n_head=2, n_positions=96, key_dim=8,
        cross_irreps="2x0e + 2x1o + 1x1e + 1x2e",
        vec_readout=True, vec_token_id=tok.vec_id, **kw,
    )


def test_tokenizer_encodes_single_vec_token_and_preserves_specials():
    tok = CharTokenizer()
    ids = tok.encode(f"F={VEC_TOKEN}", "other")
    assert ids[-1] == tok.vec_id
    assert tok.decode(ids) == f"F={VEC_TOKEN}"
    assert (tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id) == (0, 1, 2, 3)


def test_vec_readout_rejected_without_xattn_stream():
    tok = CharTokenizer()
    for bad in ("text", "prefix"):
        with pytest.raises(ValueError, match="vec_readout"):
            YLMCausalLM(make_vec_cfg(bad, tok))


@pytest.mark.parametrize("mode", ["xattnA", "xattnB", "xattnC"])
@pytest.mark.parametrize("reflection", [False, True])
def test_logits_stay_invariant_while_vec_rotates_o3(mode, reflection):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        seed_everything(7)
        tok = CharTokenizer()
        model = YLMCausalLM(make_vec_cfg(mode, tok)).double().eval()
        for adapter in model.adapters:
            adapter.gate.data.fill_(0.6)
        geom = water_geom()
        R = (-o3.rand_matrix() if reflection else o3.rand_matrix()).double()
        moved = dict(geom, pos=np.asarray(geom["pos"], dtype=np.float64) @ R.numpy().T + [3.0, -2.0, 1.5])

        ex_a = Example([Seg("force on O="), Seg(VEC_TOKEN, "answer")], geom)
        ex_b = Example([Seg("force on O="), Seg(VEC_TOKEN, "answer")], moved)
        ba = collate([ex_a], tok, 40, loss_kinds=["answer"], dtype=torch.float64)
        bb = collate([ex_b], tok, 40, loss_kinds=["answer"], dtype=torch.float64)

        target = torch.tensor([[0.25, -0.4, 0.15]], dtype=torch.float64)
        out_a = model(**model_inputs(ba), vec_targets=target)
        out_b = model(**model_inputs(bb), vec_targets=target @ R.T)

        # Text logits and combined LM + equivariant vector MSE loss remain O(3)-invariant.
        torch.testing.assert_close(out_a["logits"], out_b["logits"], atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(out_b["vec"], out_a["vec"] @ R.T, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(out_a["vec_loss"], out_b["vec_loss"], atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(out_a["loss"], out_b["loss"], atol=1e-7, rtol=1e-7)
    finally:
        torch.set_default_dtype(previous)


def test_joint_loss_gradients_and_decode_with_vectors():
    seed_everything(15)
    tok = CharTokenizer()
    model = YLMCausalLM(make_vec_cfg("xattnA", tok))
    for adapter in model.adapters:
        adapter.gate.data.fill_(0.5)
    geom = water_geom()
    ex = Example([Seg("dipole="), Seg(VEC_TOKEN, "answer")], geom)
    batch = collate([ex], tok, 32, loss_kinds=["answer"])
    target = torch.tensor([[0.1, -0.2, 0.3]])
    out = model(**model_inputs(batch), vec_targets=target)
    out["loss"].backward()
    for p in [model.vec_head.weight, model.vec_gate.weight, model.adapters[0].cross.v.weight]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0

    # Force greedy decoding to emit <vec> while keeping token embeddings and hidden states intact.
    forced_head = torch.nn.Linear(model.cfg.n_embd, tok.vocab_size, bias=True)
    with torch.no_grad():
        forced_head.weight.zero_()
        forced_head.bias.zero_()
        forced_head.bias[tok.vec_id] = 50.0
    model.lm.set_output_embeddings(forced_head)

    R = -o3.rand_matrix().float()
    moved = dict(geom, pos=np.asarray(geom["pos"], dtype=np.float32) @ R.numpy().T)
    p1 = [Example([Seg("dipole=")], geom)]
    p2 = [Example([Seg("dipole=")], moved)]
    d1 = decode_with_vectors(model, tok, p1, max_new=1)[0]
    d2 = decode_with_vectors(model, tok, p2, max_new=1)[0]
    assert d1["text"] == VEC_TOKEN and d2["text"] == VEC_TOKEN
    v1 = torch.tensor(d1["vectors"][0])
    v2 = torch.tensor(d2["vectors"][0])
    assert v1.norm() > 1e-4
    torch.testing.assert_close(v2, v1 @ R.T, atol=2e-5, rtol=2e-5)

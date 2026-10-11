"""Rotate the molecule -> the answer must rotate the same way. Move it -> nothing changes.

Run:  python -m pytest tests -q
"""
import torch
from e3nn import o3

from src.encoder import Encoder
from src.model import YLM

torch.set_default_dtype(torch.float64)  # float64 so the error is ~1e-12, easy to check


def fake_batch(B=3, N=7, T=5, d=16):
    torch.manual_seed(0)
    Z = torch.randint(1, 10, (B, N))
    pos = torch.randn(B, N, 3) * 3.0
    atom_mask = torch.ones(B, N, dtype=torch.bool)
    atom_mask[0, -2:] = False                     # molecule 0 has only 5 atoms (padding)
    text = torch.randn(B, T, d)
    text_mask = torch.ones(B, T, dtype=torch.bool)
    task = torch.tensor([0, 1, 1])
    return Z, pos, atom_mask, text, text_mask, task


def test_encoder_equivariant():
    Z, pos, mask, *_ = fake_batch()
    enc = Encoder()
    R = o3.rand_matrix()
    D = enc.irreps_out.D_from_matrix(R)
    a = enc(Z, pos @ R.T, mask)                   # rotate input
    b = enc(Z, pos, mask) @ D.T                   # rotate output
    assert (a - b).abs().max() < 1e-9


def test_model_equivariant_and_translation_invariant():
    Z, pos, mask, text, tmask, task = fake_batch()
    for mode in ["text", "taskid"]:
        model = YLM(text_dim=16, text_mode=mode)
        R = o3.rand_matrix()
        y, attn = model(Z, pos, mask, text, tmask, task)
        y_rot, attn_rot = model(Z, pos @ R.T, mask, text, tmask, task)
        y_move, _ = model(Z, pos + torch.tensor([5.0, -2.0, 1.0]), mask, text, tmask, task)
        assert (y_rot - y @ R.T).abs().max() < 1e-9, mode      # vector rotates
        assert (attn_rot - attn).abs().max() < 1e-9, mode      # attention does not change
        assert (y_move - y).abs().max() < 1e-9, mode           # translation does nothing
        assert attn[0, -2:].abs().max() == 0                   # padding atoms get no attention


def test_pipeline_and_compositional_models_equivariant():
    from src.baselines import PipelineForce, ScalarCollapse
    from src.option_bc import CompModel
    Z, pos, mask, text, tmask, task = fake_batch()
    R = o3.rand_matrix()
    shift = torch.tensor([5.0, -2.0, 1.0])
    atom = torch.tensor([-1, 2, 3])
    cases = {
        "pipeline": (PipelineForce(), (task, atom)),
        "bc": (CompModel(text_dim=16), ()),
    }
    for name, (model, extra) in cases.items():
        args = (Z, pos, mask, text, tmask) + extra
        y, _ = model(*args)
        y_rot, _ = model(Z, pos @ R.T, mask, text, tmask, *extra)
        y_move, _ = model(Z, pos + shift, mask, text, tmask, *extra)
        assert (y_rot - y @ R.T).abs().max() < 1e-8, name
        assert (y_move - y).abs().max() < 1e-8, name
    # the scalar-collapse baseline is NOT equivariant: its output must not follow the rotation
    sc = ScalarCollapse(text_dim=16)
    y, _ = sc(Z, pos, mask, text, tmask, task)
    y_rot, _ = sc(Z, pos @ R.T, mask, text, tmask, task)
    assert (y_rot - y).abs().max() < 1e-8                      # invariant: same output after rotation
    assert (y_rot - y @ R.T).abs().max() > 1e-4                # so it cannot be equivariant

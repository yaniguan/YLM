"""Tests for GroupProgModel, shipped generative <vec> checkpoint, and measured scaling/interface studies."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import src  # noqa: F401
from e3nn import o3
from src.prog import GroupProgModel
from src.tasks.tokenizer import CharTokenizer
from src.toolkit.ingest import StructureFrame
from src.toolkit.probe import YLMProbe
from src.ylm_lm import YLMCausalLM, YLMConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def float64_default():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def test_group_prog_model_o3_equivariance_and_permutation(float64_default):
    g = torch.Generator().manual_seed(7)
    B, N, T, d = 3, 6, 5, 16
    Z = torch.randint(1, 10, (B, N), generator=g)
    pos = torch.randn((B, N, 3), generator=g, dtype=torch.float64) * 2.5
    mask = torch.ones((B, N), dtype=torch.bool)
    mask[0, -1] = False
    text = torch.randn((B, T, d), generator=g, dtype=torch.float64)
    tmask = torch.ones((B, T), dtype=torch.bool)

    model = GroupProgModel(
        text_dim=d,
        hidden_irreps="8x0e + 2x0o + 4x1o + 2x1e + 2x2e",
        n_layers=2,
        d=32,
        key=16,
    )
    # Bias task head toward the 4th operator ("group") on at least one sample
    with torch.no_grad():
        model.task_head[-1].bias.copy_(torch.tensor([-2.0, -2.0, -2.0, 3.0], dtype=torch.float64))

    for training in (True, False):
        model.train(training)
        R = -o3.rand_matrix()
        assert torch.det(R) < 0
        y0, aux0 = model(Z, pos, mask, text, tmask)
        yr, _ = model(Z, pos @ R.T, mask, text, tmask)
        assert "group_logits" in aux0
        assert (yr - y0 @ R.T).abs().max() < 1e-8

        perm = torch.tensor([2, 0, 1, 4, 3, 5])
        yp, _ = model(Z[:, perm], pos[:, perm], mask[:, perm], text, tmask)
        assert (yp - y0).abs().max() < 1e-8


def test_load_v0_encoder_into_ylm_causal_lm():
    tok = CharTokenizer()
    cfg = YLMConfig(
        mode="xattnA",
        lm="tiny",
        vocab_size=tok.vocab_size,
        n_positions=128,
        n_layer=2,
        n_embd=64,
        n_head=2,
        encoder="small",
        n_elements=100,
        vec_readout=True,
        vec_token_id=tok.vec_id,
    )
    model = YLMCausalLM(cfg)
    scale = model.load_v0_encoder(str(REPO_ROOT / "checkpoints" / "v0" / "ylm_minilm_s0.pt"), freeze=True)
    assert scale.shape == (2,)
    assert not any(p.requires_grad for p in model.irreps.encoder.parameters())


def test_shipped_generative_v0_checkpoint_and_o3_probe():
    ckpt_dir = REPO_ROOT / "checkpoints" / "generative_v0"
    assert (ckpt_dir / "model.pt").exists()
    assert (ckpt_dir / "config.json").exists()
    res_json = json.loads((ckpt_dir / "results.json").read_text(encoding="utf-8"))
    assert res_json["status"] == "MEASURED"
    assert res_json["test"]["vec_emission_rate"] >= 0.95
    assert res_json["test"]["dipole"]["cosine"] > 0.95
    assert res_json["test"]["force"]["cosine"] > 0.20

    probe = YLMProbe.from_checkpoint(ckpt_dir, device="cpu")
    water = StructureFrame(
        Z=[8, 1, 1],
        pos_bohr=[[0.0, 0.0, 0.0], [1.81, 0.0, 0.0], [-0.45, 1.75, 0.0]],
        label="water",
    )
    out = probe.ask(water, "What is the dipole moment of the molecule?", verify_o3=True)
    assert out.engine == "causal_lm"
    assert "<vec>" in out.answer
    assert len(out.vectors) >= 1
    assert out.vectors[0].unit == "Debye"
    assert out.o3_check is not None and out.o3_check.passed


def test_measured_interface_and_scaling_artifacts_present():
    ig = json.loads((REPO_ROOT / "results" / "interface_generalization" / "results.json").read_text(encoding="utf-8"))
    assert ig["status"] == "MEASURED"
    assert len(ig["seeds"]) >= 3
    # Rule parser succeeds on rigid templates, drops on OOD synonyms & multi-atom groups; GroupProg generalizes
    assert ig["interfaces"]["rule_exec"]["rigid_single"]["parse_rate"] > 0.90
    assert ig["interfaces"]["rule_exec"]["ood_synonym_single"]["parse_rate"] == 0.0
    assert ig["interfaces"]["rule_exec"]["group_ood"]["parse_rate"] == 0.0
    assert ig["interfaces"]["group_prog"]["ood_synonym_single"]["cosine_mean"] > 0.75
    assert ig["interfaces"]["group_prog"]["group_ood"]["cosine_mean"] > ig["interfaces"]["optA_cont"]["group_ood"]["cosine_mean"]

    sc = json.loads((REPO_ROOT / "results" / "scaling" / "results.json").read_text(encoding="utf-8"))
    assert sc["status"] == "MEASURED"
    fc_p = sc["models"]["Option C (fully_connected)"]["cross_attn_params"]
    cw_p = sc["models"]["Option C (channelwise)"]["cross_attn_params"]
    assert fc_p > 10 * cw_p

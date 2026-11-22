"""Regression tests for full-repository codebase audit fixes."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from e3nn import o3

from labledger import Ledger
from labledger.export import to_training
from recipes.coscientist_posttrain import collate as recipe_collate
from recipes.coscientist_posttrain import schema as recipe_schema
from recipes.coscientist_posttrain import verifiers as recipe_verifiers
from src.baselines import ScalarCollapse
from src.irreps_to_text import IrrepsToText
from src.tasks.tokenizer import CharTokenizer
from src.token_quality import CTX, train as train_tiny_gpt
from src.toolkit.ingest import StructureFrame, parse_poscar
from src.toolkit.probe import ProbeResult, YLMProbe
from src.toolkit.viz import render_html_report
from src.ylm_lm import YLMCausalLM, YLMConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(rel_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_causal_lm_probe_without_vec_head():
    tok = CharTokenizer()
    cfg = YLMConfig(
        mode="xattnA",
        lm="tiny",
        vocab_size=tok.vocab_size,
        n_positions=128,
        n_layer=2,
        n_embd=64,
        n_head=2,
        bos_token_id=tok.bos_id,
        eos_token_id=tok.eos_id,
        encoder="tiny",
        vec_readout=False,
    )
    model = YLMCausalLM(cfg)
    assert model.vec_head is None

    probe = YLMProbe.from_causal_lm(model, tokenizer=tok, device="cpu")
    frame = StructureFrame(
        Z=[8, 1, 1],
        pos_bohr=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.4, 1.7, 0.0]],
        label="water",
    )
    res = probe.ask(frame, "What is the dipole?", max_new=4, verify_o3=False)
    assert res.engine == "causal_lm"
    assert isinstance(res.answer, str)
    assert res.vectors == []


def test_scalar_collapse_padding_invariance():
    torch.manual_seed(0)
    model = ScalarCollapse(
        text_dim=16,
        hidden_irreps="8x0e + 2x0o + 4x1o + 2x1e + 2x2e",
        n_layers=2,
        d=32,
    ).eval()

    Z = torch.tensor([[8, 1, 1]])
    pos = torch.tensor([[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.4, 1.7, 0.0]]])
    atom_mask = torch.ones(1, 3, dtype=torch.bool)
    text_short = torch.randn(1, 3, 16)
    mask_short = torch.ones(1, 3, dtype=torch.bool)

    text_padded = torch.cat([text_short, torch.randn(1, 5, 16) * 100.0], dim=1)
    mask_padded = torch.tensor([[True, True, True, False, False, False, False, False]])
    task = torch.tensor([0])

    with torch.no_grad():
        out_short, attn_short = model(Z, pos, atom_mask, text_short, mask_short, task)
        out_padded, attn_padded = model(Z, pos, atom_mask, text_padded, mask_padded, task)

    assert torch.allclose(out_short, out_padded, atol=1e-5)
    assert torch.allclose(attn_short, attn_padded, atol=1e-5)


def test_token_quality_train_exact_min_length():
    text = ("A" * CTX) + "Z"
    assert len(text) == CTX + 1
    model, losses = train_tiny_gpt(text, seed=0, steps=2, batch=2)
    assert len(losses) == 2
    assert all(math.isfinite(x) for x in losses)
    assert model is not None

    with pytest.raises(ValueError):
        train_tiny_gpt("A" * CTX, seed=0, steps=1, batch=1)


def test_labledger_export_compatible_with_recipe_schema_collate_and_opsd(tmp_path: Path):
    meta = {
        "origin": "real",
        "license": "own",
        "agent": {"type": "human", "id": "p:1"},
        "recorded_utc": "2026-01-01T00:00:00Z",
    }
    ledger = Ledger(str(tmp_path / "run"))
    ledger.add("project", "prj:1", {"title": "P", "goal": "G"}, **meta)
    ledger.add("campaign", "cmp:1", {"title": "C", "hypothesis": "H"}, parent_id="prj:1", **meta)
    ledger.add(
        "experiment",
        "exp:1",
        {"title": "E", "protocol_id": "proto:1", "status": "completed", "started_utc": "2026-01-01T00:00:00Z"},
        parent_id="cmp:1",
        **meta,
    )
    ledger.add("step", "stp:1", {"index": 1, "action": "sonicate", "status": "ok"}, parent_id="exp:1", **meta)
    ledger.add(
        "observation",
        "obs:sim1",
        {
            "modality": "sim",
            "t_utc": "2026-01-01T00:01:00Z",
            "detail": {"engine": "wignerflow", "system": "Ar", "values": {"T_K": 120.0}},
        },
        parent_id="stp:1",
        **dict(meta, origin="simulated"),
    )
    ledger.add(
        "observation",
        "obs:aud1",
        {
            "modality": "audio",
            "instrument_id": "mic:1",
            "t_utc": "2026-01-01T00:02:00Z",
            "failure_tags": ["acoustic_anomaly"],
            "detail": {
                "schema": "lab-ears/audio-event/v1",
                "detector": "cavitation",
                "score": 0.92,
                "threshold": 0.80,
                "duration_s": 2.0,
                "sample_rate_hz": 16000,
                "mic": "bench",
                "raw_audio_kept": False,
                "speech_band_dropped": True,
            },
        },
        parent_id="stp:1",
        **meta,
    )

    records = to_training(ledger)
    assert len(records) == 1
    rec = recipe_schema.validate(records[0])
    assert rec["sims"][0]["tag"] == "SIMULATED"
    assert rec["audios"][0]["instrument_id"] == "mic:1"

    tok = recipe_collate.ByteTokenizer()
    rendered = recipe_collate.render(rec, tok, k=4)
    decoded = tok.decode(rendered["ids"])
    assert "[sim 0: Ar, T_K=120]" in decoded
    assert "[audio 0: mic:1, bench mic, 2 s, score=0.92 vs threshold=0.80, flagged]" in decoded

    student = recipe_verifiers.opsd_student_record(
        rec,
        "All steps completed without error.",
    )
    assert "audios" in student
    recipe_schema.validate(student)


def test_poscar_negative_volume_scale_factor():
    poscar = "\n".join([
        "Cubic cell with target volume 27 A^3",
        "-27.0",
        "2.0 0.0 0.0",
        "0.0 2.0 0.0",
        "0.0 0.0 2.0",
        "Si",
        "1",
        "Direct",
        "0.25 0.25 0.25",
    ])
    frame = parse_poscar(poscar)
    vol = abs(float(np.linalg.det(frame.cell_angstrom)))
    assert math.isclose(vol, 27.0, rel_tol=1e-6)
    assert np.allclose(frame.pos_angstrom[0], [0.75, 0.75, 0.75], atol=1e-6)


def test_viz_html_report_escapes_script_close_tag():
    frame = StructureFrame(Z=[1, 1], pos_bohr=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], label="h2")
    res = ProbeResult(
        engine="v0",
        question="Safe? </script><script>window.__xss=1</script>",
        answer="No </script> breakout",
        vectors=[],
        tokens=["[CLS]", "Safe", "[SEP]"],
        atom_attention=[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        pooled_atom_attention=[0.5, 0.5],
        channel_norms={"0e": [1.0, 1.0, 1.0]},
        frame=frame,
    )
    html_doc = render_html_report(res)
    assert html_doc.count("</script>") == 1
    assert "<\\/script>" in html_doc


def test_analyze_profiling_verdict_handles_missing_gpu_util():
    mod = _load_script("scripts/analyze_profiling.py", "analyze_profiling_test")
    first_elem = next(iter(mod.ELEMENTWISE))
    run = {
        "key_averages": {
            "self_cuda_total_ms": 100.0,
            "self_cpu_total_ms": 20.0,
            "top_cuda": [{"name": first_elem, "self_cuda_ms": 60.0, "self_cpu_ms": 5.0, "calls": 10}],
            "top_cpu": [],
            "cudaLaunchKernel_calls": 10,
        }
    }
    bound, evidence = mod.verdict(run)
    assert "memory-bandwidth bound" in bound
    assert any("unknown" in item for item in evidence)


def test_irreps_to_text_handles_empty_atom_mask():
    cross = IrrepsToText(
        text_dim=16,
        atom_irreps=o3.Irreps("4x0e + 2x1o"),
        out_irreps=o3.Irreps("4x0e + 2x1o"),
        key_dim=16,
        option="A",
    )
    h = torch.randn(2, 3, 16)
    x = torch.randn(2, 4, 10)
    atom_mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
    e, attn = cross(h, x, atom_mask)
    assert torch.isfinite(e).all()
    assert torch.isfinite(attn).all()
    assert torch.all(e[1] == 0)
    assert torch.all(attn[1] == 0)

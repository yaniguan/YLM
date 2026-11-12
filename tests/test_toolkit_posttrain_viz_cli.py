"""Tests for 3-tier physics post-training (SFT+DPO), 3D HTML reports, Studio server, CLI, and HF exporter."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pickle
import threading
import urllib.request

import numpy as np
from e3nn import o3
import torch

from src.toolkit.cli import main as cli_main
from src.toolkit.domains import scan_md_trajectory
from src.toolkit.hf_export import export_tensorqa_hf
from src.toolkit.ingest import load_structure
from src.toolkit.posttrain import analytical_forces_and_dipole, run_posttrain
from src.toolkit.probe import YLMProbe
from src.toolkit.server import create_studio_server

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_analytical_labeler_o3_equivariance_and_posttrain_roundtrip(tmp_path: Path):
    frame = load_structure(REPO_ROOT / "examples" / "water.xyz")
    f0, d0 = analytical_forces_and_dipole(frame)
    # Net internal force on an isolated molecule must sum to zero by Newton's 3rd law
    np.testing.assert_allclose(f0.sum(axis=0), np.zeros(3), atol=1e-12)

    with torch.random.fork_rng():
        torch.manual_seed(7)
        R = -o3.rand_matrix(dtype=torch.float64).numpy()  # improper O(3)
    moved = frame.transformed(R, np.array([4.5, -2.2, 1.8]))
    f_rot, d_rot = analytical_forces_and_dipole(moved)
    np.testing.assert_allclose(f_rot, f0 @ R.T, atol=1e-11)
    np.testing.assert_allclose(d_rot, d0 @ R.T, atol=1e-11)

    run_dir = tmp_path / "pt_run"
    summary = run_posttrain(
        REPO_ROOT / "examples" / "proton_transfer_traj.extxyz",
        run_dir,
        backend="analytical",
        mode="xattnC",
        c_tp_mode="channelwise",
        lm="tiny",
        steps=8,
        dpo_steps=2,
        lr=3e-3,
        freeze_lm=True,
        n_layer=2,
        n_embd=48,
        n_head=4,
    )
    assert summary["n_sft_records"] >= 4
    assert summary["final_loss"] < summary["initial_loss"]
    assert len(summary["dpo_losses"]) == 2
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "config.json").exists()

    reloaded = YLMProbe.from_checkpoint(run_dir)
    assert reloaded.engine == "causal_lm"
    chk = reloaded.verify_equivariance(REPO_ROOT / "examples" / "water.xyz", "Q: Force on O(2H)? A: ")
    assert chk.passed


def test_html_reports_and_studio_http_server(tmp_path: Path):
    probe = YLMProbe.default()
    res = probe.ask(REPO_ROOT / "examples" / "water.xyz", "What is the dipole moment?", verify_o3=True)
    html_file = tmp_path / "water_report.html"
    html_text = res.to_html(html_file, title="Water Probe Test")
    assert html_file.exists()
    assert "Water Probe Test" in html_text
    assert "fallbackCanvas" in html_text
    assert "O(3) Equivariant PASS" in html_text

    scan = scan_md_trajectory(REPO_ROOT / "examples" / "proton_transfer_traj.extxyz", probe=probe)
    traj_html = scan.to_html(tmp_path / "traj_report.html")
    assert "frameSlider" in traj_html

    srv = create_studio_server(probe=probe, host="127.0.0.1", port=0)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as r_idx:
            assert r_idx.status == 200
            assert "YLM Directional Chemistry Studio" in r_idx.read().decode("utf-8")

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health") as r_health:
            health = json.loads(r_health.read().decode("utf-8"))
            assert health["status"] == "ok"

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/ask",
            data=json.dumps({
                "structure": "O",
                "domain": "general",
                "question": "What is the dipole moment of water?",
                "verify_o3": True,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r_ask:
            payload = json.loads(r_ask.read().decode("utf-8"))
            assert payload["o3_check"]["passed"] is True
            assert "fallbackCanvas" in payload["html"]
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=3)


def test_cli_subcommands_and_hf_tensorqa_exporter(tmp_path: Path):
    out_json = tmp_path / "ask.json"
    out_html = tmp_path / "ask.html"
    rc = cli_main([
        "ask",
        str(REPO_ROOT / "examples" / "water.xyz"),
        "What is the force on the oxygen atom?",
        "--verify-o3",
        "--json", str(out_json),
        "--html", str(out_html),
    ])
    assert rc == 0
    assert out_json.exists() and out_html.exists()

    rc_ver = cli_main(["verify-o3", str(REPO_ROOT / "examples" / "water.xyz")])
    assert rc_ver == 0

    # Create synthetic mini TensorQA artifacts and test `ylm export-hf --anonymized`
    labels_pkl = tmp_path / "qm9_xtb.pkl"
    v0_json = tmp_path / "tensorqa_v0.json"
    comp_json = tmp_path / "tensorqa_comp.json"
    quad_pkl = tmp_path / "qm9_quad.pkl"

    labels = [
        {
            "qm9_index": 0,
            "numbers": [8, 1, 1],
            "pos": np.array([[0.0, 0.0, 0.2], [0.0, 1.4, -0.8], [0.0, -1.4, -0.8]]),
            "dipole": np.array([0.0, 0.0, -0.72]),
            "forces": np.array([[0.0, 0.0, 0.04], [0.0, 0.02, -0.02], [0.0, -0.02, -0.02]]),
        },
        {
            "qm9_index": 1,
            "numbers": [6, 8],
            "pos": np.array([[0.0, 0.0, 0.0], [2.1, 0.0, 0.0]]),
            "dipole": np.array([-0.12, 0.0, 0.0]),
            "forces": np.array([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]]),
        },
    ]
    pickle.dump(labels, labels_pkl.open("wb"))
    v0_rows = [
        {"mol": 0, "split": "train", "task": "dipole", "atom": -1, "question": "What is the dipole moment?"},
        {"mol": 0, "split": "train", "task": "force", "atom": 0, "question": "What is the force on oxygen?"},
        {"mol": 1, "split": "test", "task": "dipole", "atom": -1, "question": "Give the dipole vector."},
        {"mol": 1, "split": "test_heldout", "task": "dipole", "atom": -1, "question": "Report total dipole."},
    ]
    v0_json.write_text(json.dumps(v0_rows), encoding="utf-8")
    comp_rows = [
        {
            "mol": 1,
            "split": "test",
            "task": "diff",
            "atoms": [0, 1],
            "target": [0.10, 0.0, 0.0],
            "question": "Subtract force on oxygen from carbon.",
        }
    ]
    comp_json.write_text(json.dumps(comp_rows), encoding="utf-8")
    q_mat = np.diag([0.4, -0.1, -0.3])
    pickle.dump({0: q_mat}, quad_pkl.open("wb"))

    hf_dir = tmp_path / "hf_release"
    rc_hf = cli_main([
        "export-hf",
        "--labels", str(labels_pkl),
        "--v0", str(v0_json),
        "--comp", str(comp_json),
        "--quad", str(quad_pkl),
        "--out", str(hf_dir),
        "--anonymized",
    ])
    assert rc_hf == 0
    card = (hf_dir / "README.md").read_text(encoding="utf-8")
    assert "double-blind peer review" in card
    assert "yaniguan" not in card
    assert (hf_dir / "v0_core" / "train.jsonl").exists()
    assert (hf_dir / "v0_compositional" / "test.jsonl").exists()
    assert (hf_dir / "v0_quadrupole" / "train.jsonl").exists()

    # Dynamically load the bundled K=32 O(3) evaluation protocol and verify equivariant scoring
    spec = importlib.util.spec_from_file_location("eval_o3", hf_dir / "evaluate_o3_protocol.py")
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)
    test_records = [json.loads(ln) for ln in (hf_dir / "v0_core" / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    metrics = eval_mod.evaluate_vector_predictions(
        test_records,
        predict_fn=lambda z, pos, q: (pos[1] - pos[0]) * (-0.057142857),
        k_rotations=8,
    )
    assert metrics["eq_err"] < 1e-12
    assert metrics["cosine"] > 0.999

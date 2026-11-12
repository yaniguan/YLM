"""Tests for the 8-Law Physics Compliance Audit Engine, 3D Pose Sandbox, CLI, and 5-Tab Studio UI."""
from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from src.toolkit import (
    evaluate_pose_sandbox,
    get_builtin_presets,
    run_physics_compliance_audit,
)
from src.toolkit.cli import main as cli_main
from src.toolkit.server import create_studio_server


@pytest.mark.parametrize(
    "preset_id",
    [
        "r_fec_chiral",
        "water_c2v",
        "trans_difluoroethene_c2h",
        "lifepo4_pbc_crystal",
    ],
)
def test_eight_law_physics_compliance_audit_passes_in_float64(preset_id: str) -> None:
    report = run_physics_compliance_audit(
        frame_or_source=preset_id,
        question="What is the 3D molecular dipole moment vector?",
        precision="float64",
        n_rotations=4,
    )
    assert report["total_laws"] == 8
    assert report["passed_count"] == 8, f"Failed laws: {[l for l in report['laws'] if not l['passed']]}"
    assert report["passed_all"] is True
    assert report["v0_probe_check"]["passed"] is True

    by_id = {law["law_id"]: law for law in report["laws"]}
    for lid in (
        "L1_translation",
        "L2_so3_rotation",
        "L3_improper_parity",
        "L4_permutation",
        "L5_tensor_symmetry",
        "L6_curie_selection",
        "L8_pbc_lattice",
    ):
        assert by_id[lid]["passed"] is True
        assert by_id[lid]["ylm_error"] < 1e-8

    assert by_id["L7_conservative_mechanics"]["passed"] is True
    assert by_id["L7_conservative_mechanics"]["ylm_error"] < 5e-6

    if preset_id == "water_c2v":
        assert report["symmetry_detected"]["is_planar"] is True
    elif preset_id == "trans_difluoroethene_c2h":
        assert report["symmetry_detected"]["has_inversion_center"] is True


def test_pose_sandbox_roto_reflection_translation_and_permutation() -> None:
    presets = get_builtin_presets()
    assert len(presets) >= 10

    res = evaluate_pose_sandbox(
        "r_bcfm_chiral",
        question="Predict the parity-odd 3D chirality sign and force on bromine.",
        rx_deg=41.0,
        ry_deg=-63.0,
        rz_deg=118.0,
        reflect=True,
        translate_ang=(2.5, -1.5, 3.2),
        permute_atoms=True,
        precision="float64",
    )
    assert res["det_Q"] == -1.0
    assert len(res["scene_orig"]["atoms"]) == 5
    assert len(res["scene_transformed"]["arrows"]) == 3

    ylm_m = res["models"]["ylm"]
    assert ylm_m["scalar_0e"]["abs_err"] < 1e-9
    assert ylm_m["chiral_0o"]["abs_err"] < 1e-9
    assert abs(ylm_m["chiral_0o"]["orig"]) > 1e-6  # genuinely chiral!
    assert ylm_m["chiral_0o"]["orig"] * ylm_m["chiral_0o"]["transformed"] < 0.0  # flipped sign!
    assert ylm_m["vector_1o"]["rel_err"] < 1e-9
    assert ylm_m["tensor_0e2e"]["rel_err"] < 1e-9
    assert ylm_m["attn_routing_err"] < 1e-9

    uni_m = res["models"]["unimol"]
    assert uni_m["vector_1o"]["rel_err"] > 0.1  # invariant scalar pooling fails 1o rotation
    assert uni_m["chiral_0o"]["abs_err"] > 1e-4  # pairwise distances blind to reflection


def test_studio_five_tab_http_endpoints_and_audit_cli(tmp_path) -> None:
    out_json = tmp_path / "audit.json"
    rc = cli_main([
        "audit-physics",
        "water_c2v",
        "--rotations",
        "3",
        "--json",
        str(out_json),
    ])
    assert rc == 0
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["passed_all"] is True

    srv = create_studio_server(host="127.0.0.1", port=0)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"

    def _get_json(path: str) -> dict:
        with urllib.request.urlopen(base + path, timeout=30) as resp:
            assert resp.status == 200
            return json.loads(resp.read().decode("utf-8"))

    def _post_json(path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            assert resp.status == 200
            return json.loads(resp.read().decode("utf-8"))

    try:
        with urllib.request.urlopen(base + "/", timeout=15) as idx_resp:
            html = idx_resp.read().decode("utf-8")
        assert "YLM Directional Chemistry Studio" in html
        assert "pane-physics" in html
        assert "pane-ouroboros" in html
        assert "pane-benchmark" in html

        health = _get_json("/api/health")
        assert health["status"] == "ok"
        assert "physics_audit" in health["capabilities"]

        presets_resp = _get_json("/api/presets")
        assert "r_fec_chiral" in presets_resp["presets"]
        assert "chiral_multirank_24ep" in presets_resp["reference_benchmarks"]

        aud = _post_json("/api/physics_audit", {"structure": "r_fec_chiral", "n_rotations": 3})
        assert aud["passed_all"] is True
        assert aud["passed_count"] == 8

        sbx = _post_json("/api/pose_sandbox", {"structure": "r_fec_chiral", "reflect": True, "rx_deg": 25.0})
        assert sbx["det_Q"] == -1.0
        assert sbx["models"]["ylm"]["chiral_0o"]["abs_err"] < 1e-9

        md_scan = _post_json("/api/run_md", {"mode": "scan_zundel"})
        assert md_scan["n_frames"] >= 3
        assert "<svg" in md_scan["svg"]
        assert "fallbackCanvas" in md_scan["html"]

        md_live = _post_json("/api/run_md", {"mode": "live_md", "steps": 8, "temperature_k": 320.0})
        assert md_live["n_frames"] == 9
        assert "<svg" in md_live["svg"]

        ob = _post_json("/api/ouroboros", {"smiles": "O=C1OC[C@H](F)O1"})
        assert ob["stereo_ok"] is True
        assert ob["enantiomer_test"]["ylm_sum_residual"] < 1e-10

        bench = _post_json("/api/eval_benchmark", {"epochs": 1, "eval_rotations": 2})
        assert len(bench["models"]) == 4
        ylm_row = [r for r in bench["models"] if "YLM" in r["model"]][0]
        assert ylm_row["vector_eq_err"] < 1e-5
    finally:
        srv.shutdown()
        srv.server_close()


def test_reference_benchmark_comes_from_committed_results_file():
    import json

    from src.toolkit.server import _REFERENCE_RESULTS_FILE, _load_reference_benchmarks

    assert _REFERENCE_RESULTS_FILE.exists()
    saved = json.loads(_REFERENCE_RESULTS_FILE.read_text(encoding="utf-8"))
    ref = _load_reference_benchmarks()["chiral_multirank_24ep"]["models"]
    assert [m["model"] for m in ref] == [m["model"] for m in saved["models"]]
    for row, m in zip(ref, saved["models"]):
        assert row["chiral_acc"] == m["test"]["chiral_sign_acc"]
        assert row["tensor_cos"] == m["test"]["tensor_cos"]

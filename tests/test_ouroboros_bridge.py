"""Unit tests for the `yaniguan/ouroboros-ocsr` <-> `YLM` cross-repo linkage (`src/toolkit/ouroboros_bridge.py`)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.toolkit.cli import main as cli_main
from src.toolkit.ouroboros_bridge import (
    compute_signed_chiral_volumes,
    embed_stereo_smiles,
    load_ouroboros_forcefield_bundle,
    load_ouroboros_predictions_csv,
    ouroboros_frames_to_universe_records,
    probe_ouroboros_structures,
    recognize_image_with_ouroboros,
)


def _rotation_matrix_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def test_signed_chiral_volume_so3_invariance_and_enantiomer_inversion():
    r_frame = embed_stereo_smiles("F[C@H](Cl)Br")
    s_frame = embed_stereo_smiles("F[C@@H](Cl)Br")

    v_r = compute_signed_chiral_volumes(r_frame)
    v_s = compute_signed_chiral_volumes(s_frame)
    assert len(v_r) == 1 and len(v_s) == 1
    # Opposite enantiomers MUST have opposite `0o` pseudoscalar chiral volume signs!
    assert v_r[0]["handedness_sign"] == -v_s[0]["handedness_sign"]
    assert v_r[0]["signed_volume_ang3"] == pytest.approx(-v_s[0]["signed_volume_ang3"], rel=1e-2)

    # Proper SO(3) rotation (det=+1) keeps `0o` chiral volume strictly invariant
    R_proper = _rotation_matrix_z(1.13)
    r_rot = r_frame.transformed(R_proper)
    v_rot = compute_signed_chiral_volumes(r_rot)
    assert v_rot[0]["signed_volume_ang3"] == pytest.approx(v_r[0]["signed_volume_ang3"], rel=1e-6)

    # Improper O(3) reflection (det=-1) flips `0o` chiral volume sign cleanly!
    R_improper = -R_proper
    r_ref = r_frame.transformed(R_improper)
    v_ref = compute_signed_chiral_volumes(r_ref)
    assert v_ref[0]["signed_volume_ang3"] == pytest.approx(-v_r[0]["signed_volume_ang3"], rel=1e-6)


def test_ouroboros_extxyz_manifest_join_and_backslash_recovery(tmp_path: Path):
    extxyz_p = tmp_path / "batch.extxyz"
    manifest_p = tmp_path / "batch.manifest.jsonl"

    # Frame 0: Z-1,2-difluoroethene where ASE extxyz mangled `F/C=C\F` -> `F/C=CF`, recovered from manifest!
    # Frame 1: Inverted conformer (`stereo_ok=F`) that MUST be filtered out when `only_stereo_ok=True`.
    extxyz_text = """6
Properties=species:S:1:pos:R:3:mace_forces:R:3 mace_energy=-268.4250 source_id="mol_z" conf_id=0 smiles="F/C=CF" stereo_ok=T
F   -1.820000  0.680000  0.000000   0.120000 -0.040000  0.000000
C   -0.660000  0.000000  0.000000  -0.080000  0.020000  0.000000
C    0.660000  0.000000  0.000000   0.080000  0.020000  0.000000
F    1.820000  0.680000  0.000000  -0.120000 -0.040000  0.000000
H   -0.680000 -1.080000  0.000000   0.000000  0.020000  0.000000
H    0.680000 -1.080000  0.000000   0.000000  0.020000  0.000000
5
Properties=species:S:1:pos:R:3:mace_forces:R:3 mace_energy=-812.1100 source_id="mol_bad" conf_id=0 smiles="F[C@H](Cl)Br" stereo_ok=F
C    0.000000  0.000000  0.000000   0.010000  0.000000  0.000000
F    0.000000  0.000000 -1.360000  -0.010000  0.000000  0.020000
Cl  -1.670000  0.000000  0.560000   0.000000  0.010000 -0.010000
Br   0.910000 -1.580000  0.640000   0.000000 -0.010000 -0.010000
H    0.520000  0.890000  0.360000   0.000000  0.000000  0.000000
"""
    extxyz_p.write_text(extxyz_text, encoding="utf-8")
    manifest_rows = [
        {
            "id": "mol_z",
            "status": "ok",
            "smiles": r"F/C=C\F",
            "inchikey": "WFLOTYSKFUPZQB-UPHRSURJSA-N",
            "accepted": True,
            "margin": 0.91,
            "conformers": [{"conf_id": 0, "stereo_ok": True, "mace_energy": -268.4250}],
        },
        {
            "id": "mol_bad",
            "status": "ok",
            "smiles": "F[C@H](Cl)Br",
            "inchikey": "XJGPCKZQYXYJAC-SCSAIBSYSA-N",
            "accepted": True,
            "margin": 0.75,
            "conformers": [{"conf_id": 0, "stereo_ok": False, "mace_energy": -812.1100}],
        },
    ]
    manifest_p.write_text("\n".join(json.dumps(r) for r in manifest_rows) + "\n", encoding="utf-8")

    # Auto-discovers sibling `.manifest.jsonl`, filters `stereo_ok=False`, and restores `F/C=C\F`!
    frames = load_ouroboros_forcefield_bundle(extxyz_p, only_stereo_ok=True)
    assert len(frames) == 1
    fr = frames[0]
    assert fr.metadata["smiles"] == r"F/C=C\F"
    assert fr.metadata["energy_ev"] == pytest.approx(-268.4250)
    assert fr.forces_ev_ang is not None and fr.forces_ev_ang.shape == (6, 3)
    assert fr.forces_ev_ang[0, 0] == pytest.approx(0.12)

    # Convert to UniverseRecord rows (C2v cis-F/C=C\F has 3 global tasks: scalar, dipole, tensor)
    recs = ouroboros_frames_to_universe_records(frames, lot="wb97m")
    assert len(recs) >= 3
    sc_recs = [r for r in recs if r.task_kind == "scalar"]
    assert sc_recs and sc_recs[0].scalar_target == pytest.approx(-268.4250)


def test_ouroboros_predictions_csv_abstention_and_cli_integration(tmp_path: Path):
    csv_p = tmp_path / "predictions.csv"
    csv_p.write_text(
        "id,smiles,accepted,margin,source_image\n"
        "fec_r,O=C1OC[C@H](F)O1,true,0.95,fig_r.png\n"
        "low_conf,CCO,false,0.12,fig_blur.png\n",
        encoding="utf-8",
    )
    frames = load_ouroboros_predictions_csv(csv_p, only_accepted=True)
    assert len(frames) == 1
    assert frames[0].metadata["smiles"] == "O=C1OC[C@H](F)O1"

    out_uni = tmp_path / "uni.jsonl"
    out_json = tmp_path / "probe.json"
    out_html = tmp_path / "probe.html"
    rc = cli_main([
        "from-ouroboros",
        str(csv_p),
        "--only-accepted",
        "--verify-o3",
        "--out-universe-jsonl", str(out_uni),
        "--json", str(out_json),
        "--html", str(out_html),
    ])
    assert rc == 0
    assert out_uni.exists() and out_json.exists() and out_html.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["n_conformers_loaded"] == 1
    assert payload["results"][0]["o3_check"]["passed"] is True

    # Helpful error message when `.png` image is passed without `ouroboros-ocsr` installed
    dummy_png = tmp_path / "mol.png"
    dummy_png.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ImportError, match="ouroboros-ocsr"):
        recognize_image_with_ouroboros(dummy_png)

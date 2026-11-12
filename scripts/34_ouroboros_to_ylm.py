"""End-to-end showcase linking `yaniguan/ouroboros-ocsr` artifacts to `YLM` directional & chiral tools.

Demonstrates:
  1. Synthesizing or reading an `ouroboros-ocsr` `image_to_forcefield.py` bundle
     (`<stem>.extxyz` with `mace_energy`, `mace_forces`, `stereo_ok` + `<stem>.manifest.jsonl`
     recovering unescaped `E/Z` SMILES `F/C=C\\F` and `(R)/(S)`-FEC battery additive stereoisomers),
  2. Auditing `0o` signed chiral volumes (`V(-R) = -V(R)`) and filtering corrupted `stereo_ok=False`
     or rejected OCSR abstention conformers,
  3. Exporting leak-free `UniverseRecord` multi-task QA datasets (`0e` energy, `1o` dipole/forces,
     `0e+2e` polarizability tensor) grouped by 14-char connectivity InChIKey,
  4. Running `YLMProbe` with live `O(3)` equivariance verification and 3D HTML report generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.toolkit.ingest import symbol_for_z
from src.toolkit.mlip_md import AnalyticalReactiveForceProvider
from src.toolkit.ouroboros_bridge import (
    embed_stereo_smiles,
    load_ouroboros_forcefield_bundle,
    ouroboros_frames_to_universe_records,
    probe_ouroboros_structures,
)


def build_demo_ouroboros_bundle(out_dir: Path) -> tuple[Path, Path]:
    """Write a realistic `ouroboros-ocsr` `.extxyz` + `.manifest.jsonl` bundle on CPU."""
    out_dir.mkdir(parents=True, exist_ok=True)
    extxyz_p = out_dir / "ouroboros_demo.extxyz"
    manifest_p = out_dir / "ouroboros_demo.manifest.jsonl"
    force_fn = AnalyticalReactiveForceProvider()

    specs = [
        ("fig1_r_fec", "O=C1OC[C@H](F)O1", True, 0.94, True),
        ("fig1_s_fec", "O=C1OC[C@@H](F)O1", True, 0.92, True),
        ("fig2_z_dfe", r"F/C=C\F", True, 0.88, True),
        ("fig3_inverted_bad", "F[C@H](Cl)Br", True, 0.81, False),  # Simulated failed 3D stereo audit -> filtered!
    ]

    extxyz_blocks: list[str] = []
    manifest_lines: list[str] = []

    for src_id, smi, accepted, margin, stereo_ok in specs:
        fr = embed_stereo_smiles(smi, check_3d_stereo=False)
        pos_ang = fr.pos_angstrom
        e_ev, f_ev_ang = force_fn(fr.Z, pos_ang)
        ik = str(fr.metadata.get("inchikey", ""))

        # Simulate ASE extxyz backslash mangling on `F/C=C\F` in the header comment
        escaped_header_smi = smi.replace("\\", "")
        header = (
            f'Properties=species:S:1:pos:R:3:mace_forces:R:3 '
            f'mace_energy={e_ev:.8f} source_id="{src_id}" conf_id=0 '
            f'smiles="{escaped_header_smi}" inchikey="{ik}" '
            f'mace_model="MACE-OFF23-small" stereo_ok={"T" if stereo_ok else "F"} has_energy=T'
        )
        lines = [str(fr.n_atoms), header]
        for z, (x, y, zc), (fx, fy, fz) in zip(fr.Z, pos_ang, f_ev_ang):
            lines.append(
                f"{symbol_for_z(int(z)):2s} {x:13.6f} {y:13.6f} {zc:13.6f} {fx:13.6f} {fy:13.6f} {fz:13.6f}"
            )
        extxyz_blocks.append("\n".join(lines))

        manifest_row = {
            "id": src_id,
            "status": "ok",
            "smiles": smi,  # Authoritative unescaped SMILES preserved in manifest!
            "inchikey": ik,
            "source_image": f"patent_figures/{src_id}.png",
            "accepted": accepted,
            "margin": margin,
            "stereo_unassigned": 0,
            "conformers": [{"conf_id": 0, "stereo_ok": stereo_ok, "mace_energy": e_ev}],
        }
        manifest_lines.append(json.dumps(manifest_row))

    extxyz_p.write_text("\n".join(extxyz_blocks) + "\n", encoding="utf-8")
    manifest_p.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return extxyz_p, manifest_p


def main() -> int:
    ap = argparse.ArgumentParser(description="Ouroboros-OCSR -> YLM 3D stereo & directional bridge demo")
    ap.add_argument("--extxyz", help="Optional existing Ouroboros .extxyz file; omit to synthesize demo bundle")
    ap.add_argument("--manifest", help="Optional explicit .manifest.jsonl path")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "ouroboros_ylm_demo"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.extxyz:
        extxyz_p = Path(args.extxyz)
        manifest_p = Path(args.manifest) if args.manifest else None
    else:
        extxyz_p, manifest_p = build_demo_ouroboros_bundle(out_dir)

    frames = load_ouroboros_forcefield_bundle(extxyz_p, manifest_path=manifest_p, only_stereo_ok=True)
    universe_recs = ouroboros_frames_to_universe_records(frames, lot="wb97m")
    uni_path = out_dir / "ouroboros_universe_qa.jsonl"
    uni_path.write_text("\n".join(r.to_json() for r in universe_recs) + "\n", encoding="utf-8")

    probe_results = probe_ouroboros_structures(
        extxyz_p,
        "What is the force vector on the fluorine atom?",
        manifest_path=manifest_p,
        only_stereo_ok=True,
        verify_o3=True,
    )
    html_p = out_dir / "r_fec_probe.html"
    if probe_results:
        probe_results[0].to_html(html_p, title="Ouroboros-OCSR -> YLM: (R)-FEC Directional Force & Chirality")

    summary = {
        "extxyz": str(extxyz_p),
        "n_stereo_verified_conformers": len(frames),
        "recovered_smiles": [fr.metadata.get("smiles") for fr in frames],
        "chiral_volumes_r_vs_s": [
            {
                "label": fr.label,
                "smiles": fr.metadata.get("smiles"),
                "chiral_volumes": fr.metadata.get("chiral_volumes", []),
            }
            for fr in frames
        ],
        "n_universe_qa_records": len(universe_recs),
        "universe_jsonl": str(uni_path),
        "o3_passed_all": all(bool(r.o3_check and r.o3_check.passed) for r in probe_results),
        "html_report": str(html_p),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line tool (`ylm`) for 3D chemistry questions, physics checks, MD monitoring, and Studio UI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from src.toolkit.domains import analyze_drug, analyze_electrolyte, scan_md_trajectory
from src.toolkit.hf_export import export_tensorqa_hf
from src.toolkit.ingest import build_solvation_cluster, load_trajectory
from src.toolkit.md_hooks import LiveReactionMonitor, plot_reaction_timeline
from src.toolkit.mlip_md import MLIPTrajectoryRunner, TorchMLIPForceProvider
from src.toolkit.ouroboros_bridge import (
    load_any_ouroboros_input,
    ouroboros_frames_to_universe_records,
    probe_ouroboros_structures,
)
from src.toolkit.physics_audit import run_physics_compliance_audit
from src.toolkit.posttrain import run_posttrain
from src.toolkit.probe import YLMProbe, verify_o3_equivariance
from src.toolkit.server import create_studio_server


def _load_probe(
    ckpt: Optional[str],
    device: str,
    mace: Optional[str] = None,
    nequip: Optional[str] = None,
) -> YLMProbe:
    if mace:
        return YLMProbe.from_mace(backbone=mace, device=device)
    if nequip:
        return YLMProbe.from_nequip(backbone=nequip, device=device)
    return YLMProbe.from_checkpoint(ckpt, device=device) if ckpt else YLMProbe.default(device=device)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ylm", description="YLM O(3)-equivariant directional chemistry probe & toolkit")
    sub = p.add_subparsers(dest="command", required=True)

    # ylm ask
    p_ask = sub.add_parser("ask", help="Ask a natural-language 3D directional question on a structure or SMILES")
    p_ask.add_argument("structure", help="Path to .xyz/.extxyz/.sdf/POSCAR/.cif or SMILES string")
    p_ask.add_argument("question", help="Natural-language directional question")
    p_ask.add_argument("--domain", choices=["general", "electrolyte", "md", "drug"], default="general")
    p_ask.add_argument("--ckpt", help="Checkpoint file (.pt) or YLMCausalLM run directory")
    p_ask.add_argument("--mace", help="Mount pretrained MACE backbone (e.g. mace-mp-0-small, mace-off23-medium)")
    p_ask.add_argument("--nequip", help="Mount compiled TorchScript or Python NequIP/Allegro backbone (.pth)")
    p_ask.add_argument("--device", default="cpu")
    p_ask.add_argument("--verify-o3", action="store_true", help="Run live O(3) rotation+reflection verification")
    p_ask.add_argument("--html", help="Write standalone interactive 3D HTML report to this path")
    p_ask.add_argument("--json", help="Write structured JSON result to this path")

    # ylm solvation
    p_solv = sub.add_parser("solvation", help="Analyze electrolyte ion solvation shell & desolvation escape vector")
    p_solv.add_argument("structure", nargs="?", help="Optional solvation cluster / crystal file; omit to build from --solvents")
    p_solv.add_argument("--ion", default="Li", help="Working ion symbol when building from --solvents (default: Li)")
    p_solv.add_argument("--solvents", nargs="+", default=["O=C1OCCO1", "O=C1OCCO1", "O=C1OCCO1"],
                        help="Solvent SMILES list when no structure file is supplied")
    p_solv.add_argument("--question", help="Custom directional question")
    p_solv.add_argument("--ckpt")
    p_solv.add_argument("--device", default="cpu")
    p_solv.add_argument("--verify-o3", action="store_true")
    p_solv.add_argument("--html")
    p_solv.add_argument("--json")

    # ylm scan-traj
    p_traj = sub.add_parser("scan-traj", help="Scan a multi-frame MD trajectory (.extxyz/.xyz/.traj) with 1o motion")
    p_traj.add_argument("trajectory", help="Path to multi-frame .extxyz / .xyz / .traj")
    p_traj.add_argument("--query", default="What is the force vector on the transferring hydrogen atom?")
    p_traj.add_argument("--every", type=int, default=1)
    p_traj.add_argument("--dt-fs", type=float, default=1.0, help="MD frame timestep in femtoseconds for timeline plots")
    p_traj.add_argument("--ckpt")
    p_traj.add_argument("--mace", help="Mount pretrained MACE backbone (e.g. mace-mp-0-small, mace-off23-medium)")
    p_traj.add_argument("--nequip", help="Mount compiled TorchScript or Python NequIP/Allegro backbone (.pth)")
    p_traj.add_argument("--device", default="cpu")
    p_traj.add_argument("--verify-o3", action="store_true")
    p_traj.add_argument("--html")
    p_traj.add_argument("--plot", help="Write reaction timeline plot (.svg or .png) with frame/time markers")
    p_traj.add_argument("--json")

    # ylm drug
    p_drug = sub.add_parser("drug", help="Analyze ligand/pocket pharmacophores & steric-safe 3D substituent exit vectors")
    p_drug.add_argument("structure", help="Ligand or ligand+pocket file (.xyz/.sdf) or SMILES")
    p_drug.add_argument("--question", help="Custom directional question")
    p_drug.add_argument("--ckpt")
    p_drug.add_argument("--device", default="cpu")
    p_drug.add_argument("--verify-o3", action="store_true")
    p_drug.add_argument("--html")
    p_drug.add_argument("--json")

    # ylm posttrain
    p_pt = sub.add_parser("posttrain", help="Fine-tune YLMCausalLM + IrrepsAdapter + <vec> head with SFT and optional DPO")
    p_pt.add_argument("--input", required=True, help="Input .xyz/.extxyz or SMILES")
    p_pt.add_argument("--out", required=True, help="Output run directory")
    p_pt.add_argument("--backend", choices=["auto", "analytical", "mmff", "xtb", "extxyz"], default="auto")
    p_pt.add_argument("--mode", choices=["xattnA", "xattnB", "xattnC"], default="xattnC")
    p_pt.add_argument("--c-tp-mode", choices=["fully_connected", "channelwise"], default="channelwise")
    p_pt.add_argument("--lm", default="tiny", help="Preset ('tiny', 'small') or 'hf:<repo_id>'")
    p_pt.add_argument("--steps", type=int, default=20)
    p_pt.add_argument("--dpo-steps", type=int, default=0)
    p_pt.add_argument("--lr", type=float, default=2e-3)
    p_pt.add_argument("--freeze-lm", action="store_true")
    p_pt.add_argument("--device", default="cpu")

    # ylm verify-o3
    p_ver = sub.add_parser("verify-o3", help="Run live O(3) rotation + reflection equivariance check")
    p_ver.add_argument("structure", help="Structure path or SMILES")
    p_ver.add_argument("--question", default="What is the force vector on the oxygen atom?")
    p_ver.add_argument("--ckpt")

    # ylm audit-physics
    p_aud = sub.add_parser(
        "audit-physics",
        help="Check 8 basic physics rules (move, spin, mirror, atom order, 3x3 matrix, symmetry zeros, F=-dE/dx, crystal box)",
    )
    p_aud.add_argument(
        "structure",
        nargs="?",
        default="r_fec_chiral",
        help="Builtin preset (r_fec_chiral, water_c2v, trans_difluoroethene_c2h, lifepo4_pbc_crystal, ...) or structure/SMILES",
    )
    p_aud.add_argument("--question", default="What is the 3D molecular dipole moment vector?")
    p_aud.add_argument("--precision", choices=["float64", "float32"], default="float64")
    p_aud.add_argument("--rotations", type=int, default=8)
    p_aud.add_argument("--ckpt")
    p_aud.add_argument("--device", default="cpu")
    p_aud.add_argument("--json", help="Write full structured audit report JSON to this path")

    # ylm serve
    p_srv = sub.add_parser("serve", help="Launch local interactive YLM Directional Chemistry Studio in browser")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8765)
    p_srv.add_argument("--ckpt")
    p_srv.add_argument("--device", default="cpu")

    # ylm export-hf
    p_hf = sub.add_parser("export-hf", help="Export TensorQA JSON/pickle files into a Hugging Face Dataset directory")
    p_hf.add_argument("--labels", default="data/qm9_xtb.pkl", help="Path to qm9_xtb.pkl")
    p_hf.add_argument("--v0", default="data/tensorqa_v0.json", help="Path to tensorqa_v0.json")
    p_hf.add_argument("--out", required=True, help="Output directory for Hugging Face dataset release")
    p_hf.add_argument("--comp", help="Optional path to tensorqa_comp.json (T4 diff & T5 proj)")
    p_hf.add_argument("--quad", help="Optional path to qm9_quad.pkl (PySCF l=2 quadrupole)")
    p_hf.add_argument("--anonymized", action="store_true", help="Omit repo/author identifiers for double-blind review")

    # ylm run-md
    p_md = sub.add_parser("run-md", help="Run live reactive MLIP MD (MACE / NequIP / analytical) with reaction detection")
    p_md.add_argument("structure", help="Initial structure path (.xyz/.extxyz/.sdf/POSCAR/.cif) or SMILES")
    p_md.add_argument("--steps", type=int, default=20, help="Number of Velocity-Verlet steps")
    p_md.add_argument("--dt-fs", type=float, default=0.5, help="Timestep in femtoseconds")
    p_md.add_argument("--temp", type=float, default=300.0, help="Langevin thermostat temperature (K)")
    p_md.add_argument("--mace", help="Drive MD with MACE backbone (e.g. mace-mp-0-small, mace-off23-small)")
    p_md.add_argument("--nequip", help="Drive MD with NequIP/Allegro backbone (.pth)")
    p_md.add_argument("--ckpt", help="Optional YLM probe checkpoint for reactive-frame directional attribution")
    p_md.add_argument("--device", default="cpu")
    p_md.add_argument("--out-traj", help="Write multi-frame .extxyz trajectory with energies & forces")
    p_md.add_argument("--json", help="Write detected reaction events & history JSON")
    p_md.add_argument("--plot", help="Write SVG/PNG reaction timeline plot")
    p_md.add_argument("--html", help="Write interactive 3D HTML trajectory report")

    # ylm from-ouroboros
    p_ob = sub.add_parser(
        "from-ouroboros",
        help="Link yaniguan/ouroboros-ocsr (.extxyz, .manifest.jsonl, predictions.csv, image, or stereo SMILES) to YLM",
    )
    p_ob.add_argument("source", help="Path to Ouroboros .extxyz / .manifest.jsonl / predictions.csv / image or stereo SMILES")
    p_ob.add_argument("--question", default="What is the 3D molecular dipole vector?", help="Directional question for YLMProbe")
    p_ob.add_argument("--manifest", help="Optional explicit path to <stem>.manifest.jsonl")
    p_ob.add_argument("--ouroboros-repo", help="Optional local path to yaniguan/ouroboros-ocsr clone")
    p_ob.add_argument("--ouroboros-ckpt", help="Optional Ouroboros OCSR checkpoint when input is a 2D image")
    p_ob.add_argument("--only-accepted", action="store_true", help="Filter out rejected Arm A+C OCSR abstentions")
    p_ob.add_argument("--min-margin", type=float, help="Minimum Ouroboros confidence margin threshold")
    p_ob.add_argument("--lot", default="wb97m", help="Level-of-Theory tag when exporting UniverseRecord JSONL")
    p_ob.add_argument("--out-universe-jsonl", help="Export leak-free multi-task UniverseRecord JSONL dataset")
    p_ob.add_argument("--ckpt", help="YLM checkpoint (.pt) or YLMCausalLM run dir")
    p_ob.add_argument("--mace", help="Mount pretrained MACE backbone")
    p_ob.add_argument("--nequip", help="Mount compiled NequIP/Allegro backbone")
    p_ob.add_argument("--device", default="cpu")
    p_ob.add_argument("--verify-o3", action="store_true")
    p_ob.add_argument("--html", help="Write interactive 3D HTML report for the first conformer")
    p_ob.add_argument("--json", help="Write full structured JSON summary across all conformers")

    return p


def _emit_result(res, html_path: Optional[str], json_path: Optional[str]) -> None:
    if html_path:
        res.to_html(html_path)
    payload = res.to_dict()
    if json_path:
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "answer": payload.get("answer"),
        "vectors": payload.get("vectors"),
        "domain_summary": payload.get("domain_summary"),
        "o3_check": payload.get("o3_check"),
        "html": html_path,
    }, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "ask":
        probe = _load_probe(args.ckpt, args.device, mace=args.mace, nequip=args.nequip)
        res = probe.ask(args.structure, args.question, domain=args.domain, verify_o3=args.verify_o3)
        _emit_result(res, args.html, args.json)
        return 0

    if args.command == "solvation":
        probe = _load_probe(args.ckpt, args.device)
        target = (args.structure if args.structure
                  else build_solvation_cluster(solvent_smiles=args.solvents, ion_symbol=args.ion))
        res = analyze_electrolyte(target, probe=probe, question=args.question, verify_o3=args.verify_o3)
        _emit_result(res, args.html, args.json)
        return 0

    if args.command == "scan-traj":
        probe = _load_probe(args.ckpt, args.device, mace=args.mace, nequip=args.nequip)
        scan = scan_md_trajectory(args.trajectory, probe=probe, query=args.query,
                                  every=args.every, verify_o3=args.verify_o3)
        if args.plot:
            frames = load_trajectory(args.trajectory)
            mon = LiveReactionMonitor(probe=probe)
            for idx_f, fr in enumerate(frames):
                mon.step(idx_f, idx_f * args.dt_fs, fr.Z, fr.pos_angstrom, fr.forces_ev_ang, fr.cell_angstrom, fr.pbc)
            plot_reaction_timeline(mon.history, mon.events, args.plot)
        if args.html:
            scan.to_html(args.html)
        payload = scan.to_dict()
        if args.json:
            p = Path(args.json)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({
            "n_frames_total": scan.n_frames_total,
            "scanned_indices": scan.scanned_indices,
            "events": scan.events,
            "summary": scan.summary,
            "html": args.html,
            "plot": args.plot,
        }, indent=2))
        return 0

    if args.command == "drug":
        probe = _load_probe(args.ckpt, args.device)
        res = analyze_drug(args.structure, probe=probe, question=args.question, verify_o3=args.verify_o3)
        _emit_result(res, args.html, args.json)
        return 0

    if args.command == "posttrain":
        summary = run_posttrain(
            args.input,
            args.out,
            backend=args.backend,
            mode=args.mode,
            c_tp_mode=args.c_tp_mode,
            lm=args.lm,
            steps=args.steps,
            dpo_steps=args.dpo_steps,
            lr=args.lr,
            freeze_lm=args.freeze_lm,
            device=args.device,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "verify-o3":
        chk = verify_o3_equivariance(args.structure, args.question, checkpoint=args.ckpt)
        print(json.dumps({
            "passed": chk.passed,
            "reflection_tested": chk.reflection_tested,
            "vector_equivariance_error": chk.vector_equivariance_error,
            "invariance_error": chk.invariance_error,
        }, indent=2))
        return 0 if chk.passed else 1

    if args.command == "audit-physics":
        probe = _load_probe(args.ckpt, args.device)
        audit = run_physics_compliance_audit(
            frame_or_source=args.structure,
            question=args.question,
            precision=args.precision,
            n_rotations=args.rotations,
            probe=probe,
        )
        if args.json:
            jp = Path(args.json)
            jp.parent.mkdir(parents=True, exist_ok=True)
            jp.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(f"=== YLM 8-Law Physics Compliance Audit ({audit['preset_title']} | {audit['precision']}) ===")
        for law in audit["laws"]:
            tag = "PASS" if law["passed"] else "FAIL"
            print(f"  [{tag}] {law['name']:<72} err={law['ylm_error']:.2e} (tol={law['tolerance']:.1e})")
        v0 = audit["v0_probe_check"]
        print(
            f"  [{'PASS' if v0['passed'] else 'FAIL'}] Shipped YLMProbe ({v0['engine']}) O(3) check"
            f"                                     vec_err={v0['vector_equivariance_error']:.2e}"
        )
        print(f"Overall: {audit['passed_count']}/{audit['total_laws']} laws passed | passed_all={audit['passed_all']}")
        return 0 if audit["passed_all"] else 1

    if args.command == "serve":
        probe = _load_probe(args.ckpt, args.device)
        srv = create_studio_server(probe=probe, host=args.host, port=args.port)
        print(f"YLM Studio listening on http://{args.host}:{args.port} (Ctrl+C to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            srv.server_close()
        return 0

    if args.command == "export-hf":
        summary = export_tensorqa_hf(
            args.labels,
            args.v0,
            args.out,
            comp_questions_path=args.comp,
            quad_labels_path=args.quad,
            anonymized=args.anonymized,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "run-md":
        provider = TorchMLIPForceProvider.from_spec(mace=args.mace, nequip=args.nequip, device=args.device)
        probe = _load_probe(args.ckpt, args.device, mace=args.mace, nequip=args.nequip)
        runner = MLIPTrajectoryRunner(
            provider,
            probe=probe,
            dt_fs=args.dt_fs,
            temperature_k=args.temp,
            device=args.device,
        )
        md_res = runner.run(
            args.structure,
            steps=args.steps,
            out_extxyz=args.out_traj,
            out_events_json=args.json,
            out_plot=args.plot,
            out_html=args.html,
        )
        print(json.dumps(md_res.to_summary_dict(), indent=2))
        return 0

    if args.command == "from-ouroboros":
        probe = _load_probe(args.ckpt, args.device, mace=args.mace, nequip=args.nequip)
        frames = load_any_ouroboros_input(
            args.source,
            manifest_path=args.manifest,
            ouroboros_repo=args.ouroboros_repo,
            ouroboros_ckpt=args.ouroboros_ckpt,
            only_stereo_ok=True,
            only_accepted=args.only_accepted,
            min_margin=args.min_margin,
            device=args.device,
        )
        n_universe_rows = 0
        if args.out_universe_jsonl:
            recs = ouroboros_frames_to_universe_records(frames, lot=args.lot)
            out_u = Path(args.out_universe_jsonl)
            out_u.parent.mkdir(parents=True, exist_ok=True)
            out_u.write_text("\n".join(r.to_json() for r in recs) + ("\n" if recs else ""), encoding="utf-8")
            n_universe_rows = len(recs)

        results = probe_ouroboros_structures(
            args.source,
            args.question,
            probe=probe,
            manifest_path=args.manifest,
            ouroboros_repo=args.ouroboros_repo,
            ouroboros_ckpt=args.ouroboros_ckpt,
            only_stereo_ok=True,
            only_accepted=args.only_accepted,
            min_margin=args.min_margin,
            verify_o3=args.verify_o3,
        )
        if args.html and results:
            results[0].to_html(args.html)
        summary = {
            "n_conformers_loaded": len(results),
            "n_universe_records_exported": n_universe_rows,
            "universe_jsonl": args.out_universe_jsonl,
            "results": [r.to_dict() for r in results],
        }
        if args.json:
            jp = Path(args.json)
            jp.parent.mkdir(parents=True, exist_ok=True)
            jp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({
            "n_conformers_loaded": len(results),
            "n_universe_records_exported": n_universe_rows,
            "first_answer": results[0].answer if results else None,
            "first_domain_summary": results[0].domain_summary if results else None,
            "html": args.html,
            "universe_jsonl": args.out_universe_jsonl,
        }, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

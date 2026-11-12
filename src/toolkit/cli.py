"""Unified `ylm` CLI for out-of-the-box directional probing, domain analysis, post-training, and Web Studio."""
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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

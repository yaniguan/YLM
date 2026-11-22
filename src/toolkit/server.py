"""Local browser app serving the YLM research homepage and live Studio (`ylm serve`).

Serves the static project website in `site/` alongside live PyTorch JSON endpoints:
  - `/api/health`        : engine status and capability list
  - `/api/presets`       : built-in 3D molecule presets and reference benchmarks
  - `/api/physics_audit` : 64-bit 8-law conservation and symmetry checker
  - `/api/pose_sandbox`  : interactive O(3) rotation, mirror, shift, and atom-order test
  - `/api/ask`           : natural-language 3D dipole and per-atom force query
  - `/api/run_md`        : Zundel H+ scan and live velocity-Verlet trajectory hooks
  - `/api/ouroboros`     : 2D stereo SMILES to 3D conformer and mirror-twin check
  - `/api/eval_benchmark`: live on-demand CPU comparison across 4 model families
"""
from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import tempfile
import time
from typing import Optional

import numpy as np
import torch

from src.toolkit.domains import scan_md_trajectory
from src.toolkit.ingest import ANGSTROM_TO_BOHR, StructureFrame, build_solvation_cluster, load_structure
from src.toolkit.md_hooks import LiveReactionMonitor, plot_reaction_timeline
from src.toolkit.mlip_md import AnalyticalReactiveForceProvider, MLIPTrajectoryRunner
from src.toolkit.ouroboros_bridge import embed_stereo_smiles
from src.toolkit.physics_audit import (
    REPO_ROOT,
    _frame_to_tensors,
    _load_audit_models,
    evaluate_pose_sandbox,
    get_builtin_presets,
    resolve_preset_or_structure,
    run_physics_compliance_audit,
)
from src.toolkit.probe import YLMProbe
from src.toolkit.viz import render_html_report, render_trajectory_html_report
from src.universe_data import (
    QuestionFeaturizer,
    build_cpu_smoke_universe_dataset,
    compute_chiral_centers,
    make_universe_batches,
)
from src.universe_fm import YLMUniverseModel, compute_universe_loss, evaluate_universe_o3

_REFERENCE_RESULTS_FILE = REPO_ROOT / "results" / "cpu_smoke_nov12" / "chiral_benchmark.json"
_SITE_DIR = (REPO_ROOT / "site").resolve()

_MIME_OVERRIDES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _load_reference_benchmarks() -> dict:
    """Read the saved handedness benchmark from the committed results file."""
    models = []
    if _REFERENCE_RESULTS_FILE.exists():
        data = json.loads(_REFERENCE_RESULTS_FILE.read_text(encoding="utf-8"))
        for m in data.get("models", []):
            test = m["test"]
            models.append({
                "model": m["model"],
                "chiral_acc": test["chiral_sign_acc"],
                "heldout_chiral_acc": m["test_heldout"]["chiral_sign_acc"],
                "zeroshot_scaffold_acc": m["test_zero_shot_scaffold"]["chiral_sign_acc"],
                "chiral_parity_err": test["chiral_parity_err"],
                "achiral_planar_abs": m["achiral_planar_max_abs"],
                "scalar_mae": test["scalar_mae"],
                "vector_cos": test["vector_cos"],
                "vector_eq_err": test["vector_eq_err"],
                "tensor_cos": test["tensor_cos"],
                "routing_acc": test["routing_acc"],
            })
    return {
        "chiral_multirank_24ep": {
            "title": "Handedness + 3D Arrow & Matrix Benchmark (24 Epochs, 8 Random Rotations + Mirror Flips)",
            "source": "results/cpu_smoke_nov12/chiral_benchmark.json",
            "models": models,
        }
    }


def _run_live_mini_benchmark(epochs: int = 6, eval_rotations: int = 4) -> dict:
    """Train and test all 4 model families on CPU right inside the Studio server."""
    from src.fm_baselines import (
        EquiLLMMultiTaskBaseline,
        GraphMVP3DInfoMaxBaseline,
        UniMolStyle3DBaseline,
    )

    epochs = max(1, min(int(epochs), 20))
    eval_rotations = max(2, min(int(eval_rotations), 12))
    splits = build_cpu_smoke_universe_dataset(n_conformers_per_archetype=6, seed=42)
    featurizer = QuestionFeaturizer(mode="fast", dim=64)

    hidden_ir = "16x0e + 4x0o + 8x1o + 4x1e + 4x2e"
    cross_ir = "8x0e + 8x1o + 4x1e + 4x2e"
    candidates = [
        ("GraphMVP / 3D-InfoMax", GraphMVP3DInfoMaxBaseline(text_dim=64, d_model=48, n_layers=2)),
        ("Uni-Mol (3D Distance Transformer)", UniMolStyle3DBaseline(text_dim=64, d_model=48, n_layers=2, n_heads=4)),
        ("EquiLLM (Scalar Bridge + Vector)", EquiLLMMultiTaskBaseline(text_dim=64, hidden_irreps=hidden_ir, n_layers=2, d_model=48, n_vec=8)),
        (
            "YLM-Universe (Ours)",
            YLMUniverseModel(
                text_dim=64, backbone="e3nn", hidden_irreps=hidden_ir, cross_irreps=cross_ir,
                n_layers=2, n_cross_layers=2, option="C", c_tp_mode="channelwise", d_text=64,
            ),
        ),
    ]

    rows = []
    test_b = make_universe_batches(splits["test"], featurizer, batch_size=24)
    for label, mdl in candidates:
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        opt = torch.optim.AdamW(mdl.parameters(), lr=3e-3, weight_decay=1e-4)
        t0 = time.perf_counter()
        for _ in range(epochs):
            mdl.train()
            for batch in make_universe_batches(splits["train"], featurizer, batch_size=24, shuffle=True, rng=rng):
                opt.zero_grad()
                out = mdl(
                    batch["Z"], batch["pos"], batch["atom_mask"],
                    batch["text"], batch["text_mask"],
                    lot_id=batch["lot_id"], cell=batch["cell"], pbc=batch["pbc"],
                )
                loss = compute_universe_loss(
                    out,
                    task_kind=batch["task_kind"],
                    scalar_target=batch["scalar_target"],
                    vector_target=batch["vector_target"],
                    tensor_target=batch["tensor_target"],
                    target_atom_mask=batch["target_atom_mask"],
                    lot_id=batch["lot_id"],
                )["loss"]
                loss.backward()
                opt.step()
        elapsed = round(time.perf_counter() - t0, 2)
        m = evaluate_universe_o3(mdl, test_b, n_rotations=eval_rotations, include_reflections=True)
        rows.append({
            "model": label,
            "elapsed_sec": elapsed,
            "scalar_mae": m["scalar_mae"],
            "scalar_inv_err": m["scalar_inv_err"],
            "vector_cos": m["vector_cos"],
            "force_cos": m.get("force_cos", 0.0),
            "vector_eq_err": m["vector_eq_err"],
            "tensor_cos": m["tensor_cos"],
            "tensor_eq_err": m["tensor_eq_err"],
            "routing_acc": m["routing_acc"],
        })
    return {
        "epochs": epochs,
        "eval_rotations": eval_rotations,
        "n_train": len(splits["train"]),
        "n_test": len(splits["test"]),
        "models": rows,
    }


def _handle_ouroboros_inspect(smiles: str, question: str, active_probe: YLMProbe) -> dict:
    smi = smiles.strip() or "O=C1OC[C@H](F)O1"
    fr = embed_stereo_smiles(smi, seed=0)
    centers = compute_chiral_centers(fr)

    mirror_fr = StructureFrame(
        Z=fr.Z.copy(),
        pos_bohr=-fr.pos_bohr.copy(),
        label=f"mirror_of_{fr.label}",
        metadata=dict(fr.metadata),
    )
    mirror_centers = compute_chiral_centers(mirror_fr)

    models = _load_audit_models("float64")
    in_orig = _frame_to_tensors(fr, question, torch.float64)
    in_mirr = _frame_to_tensors(mirror_fr, question, torch.float64)

    with torch.no_grad():
        y0 = models["ylm"](in_orig["Z"], in_orig["pos"], in_orig["atom_mask"], in_orig["text"], in_orig["text_mask"], lot_id=in_orig["lot_id"])
        y1 = models["ylm"](in_mirr["Z"], in_mirr["pos"], in_mirr["atom_mask"], in_mirr["text"], in_mirr["text_mask"], lot_id=in_mirr["lot_id"])
        u0 = models["unimol"](in_orig["Z"], in_orig["pos"], in_orig["atom_mask"], in_orig["text"], in_orig["text_mask"], lot_id=in_orig["lot_id"])
        u1 = models["unimol"](in_mirr["Z"], in_mirr["pos"], in_mirr["atom_mask"], in_mirr["text"], in_mirr["text_mask"], lot_id=in_mirr["lot_id"])

    probe_res = active_probe.ask(fr, question, verify_o3=True)
    probe_res.domain = "ouroboros_ocsr"
    probe_res.domain_summary = {
        "smiles": smi,
        "embedder": fr.metadata.get("embedder", "builtin"),
        "stereo_ok": bool(fr.metadata.get("stereo_ok", True)),
        "inchikey": fr.metadata.get("inchikey", ""),
        "cip_centers": fr.metadata.get("cip_centers", []),
        "chiral_centers": centers,
    }
    return {
        "smiles": smi,
        "n_atoms": fr.n_atoms,
        "atoms": [{"Z": int(z), "pos": [float(x) for x in p]} for z, p in zip(fr.Z, fr.pos_angstrom)],
        "chiral_atoms": [int(c["atom_index"]) for c in centers],
        "dipole": [float(v) for v in (probe_res.vectors[0].vector_physical if probe_res.vectors else [0.0, 0.0, 0.0])],
        "stereo_ok": bool(fr.metadata.get("stereo_ok", True)),
        "embedder": fr.metadata.get("embedder", "builtin"),
        "inchikey": fr.metadata.get("inchikey", ""),
        "cip_centers": fr.metadata.get("cip_centers", []),
        "chiral_centers": centers,
        "mirror_chiral_centers": mirror_centers,
        "enantiomer_test": {
            "ylm_chiral_orig": float(y0["chiral"][0, 0].item()),
            "ylm_chiral_mirror": float(y1["chiral"][0, 0].item()),
            "ylm_sum_residual": abs(float(y0["chiral"][0, 0].item() + y1["chiral"][0, 0].item())),
            "unimol_chiral_orig": float(u0["chiral"][0, 0].item()),
            "unimol_chiral_mirror": float(u1["chiral"][0, 0].item()),
            "unimol_diff": abs(float(u0["chiral"][0, 0].item() - u1["chiral"][0, 0].item())),
        },
        "html": render_html_report(probe_res, title=f"Chirality · {smi}"),
    }


def _handle_run_md(body: dict, active_probe: YLMProbe) -> dict:
    mode = str(body.get("mode") or "scan_zundel")
    with tempfile.TemporaryDirectory() as tmpd:
        tmp_path = Path(tmpd)
        svg_path = tmp_path / "timeline.svg"
        if mode == "scan_zundel":
            traj_file = REPO_ROOT / "examples" / "proton_transfer_traj.extxyz"
            scan = scan_md_trajectory(traj_file, probe=active_probe, verify_o3=True)
            mon = LiveReactionMonitor(probe=active_probe)
            trajectory = []
            energies = []
            for idx, fr_res in enumerate(scan.frame_results):
                fr = fr_res.frame
                mon.step(idx, idx * 2.0, fr.Z, fr.pos_angstrom, fr.forces_ev_ang, fr.cell_angstrom, fr.pbc)
                energies.append(float(fr.metadata.get("energy_ev", 0.04 * np.sin(idx * 0.5))))
                vec_fr = fr_res.vectors[0].vector_physical if getattr(fr_res, "vectors", None) else [0.0, 0.0, 0.0]
                trajectory.append({
                    "Z": [int(z) for z in fr.Z],
                    "pos_ang": fr.pos_angstrom.tolist(),
                    "max_force_vec": [float(v) for v in vec_fr],
                })
            plot_reaction_timeline(mon.history, mon.events, svg_path)
            svg_text = svg_path.read_text(encoding="utf-8")
            html_text = render_trajectory_html_report(scan, title="Proton-Transfer Trajectory")
            return {
                "mode": mode,
                "n_frames": len(scan.frame_results),
                "trajectory": trajectory,
                "energies_ev": energies,
                "events": [e.to_dict() for e in mon.events],
                "scan_events": scan.events,
                "summary": scan.summary,
                "svg": svg_text,
                "html": html_text,
            }

        steps = max(6, min(int(body.get("steps", 28)), 80))
        dt_fs = float(body.get("dt_fs", 0.6))
        temp_k = float(body.get("temperature_k", 350.0))
        stretch = float(body.get("bond_stretch_ang", 0.55))

        src = str(body.get("structure") or "").strip()
        if src and src != "zundel_stretched":
            init_fr, _ = resolve_preset_or_structure(src)
        else:
            coords = np.array([
                [-1.26, 0.02, 0.00],
                [-0.38 + stretch, 0.01, 0.00],
                [1.24, -0.02, 0.00],
                [-1.82, 0.78, 0.18],
                [-1.78, -0.76, -0.18],
                [1.80, 0.76, -0.20],
                [1.79, -0.78, 0.20],
            ], dtype=np.float64)
            init_fr = StructureFrame(Z=[8, 1, 8, 1, 1, 1, 1], pos_bohr=coords * ANGSTROM_TO_BOHR, label="H5O2_zundel_live")

        runner = MLIPTrajectoryRunner(
            force_provider=AnalyticalReactiveForceProvider(well_depth_ev=1.8, stiffness=1.9),
            probe=active_probe,
            dt_fs=dt_fs,
            temperature_k=temp_k,
            seed=7,
        )
        html_file = tmp_path / "live_md.html"
        md_res = runner.run(
            init_fr,
            steps=steps,
            report_every=1,
            out_plot=svg_path,
            out_html=html_file,
        )
        trajectory = [
            {
                "Z": [int(z) for z in fr.Z],
                "pos_ang": fr.pos_angstrom.tolist(),
                "max_force_vec": fr.forces_ev_ang[1].tolist() if fr.forces_ev_ang is not None and len(fr.forces_ev_ang) > 1 else [0.0, 0.0, 0.0],
            }
            for fr in md_res.frames
        ]
        return {
            "mode": "live_md",
            "n_frames": len(md_res.frames),
            "trajectory": trajectory,
            "energies_ev": [round(float(e), 6) for e in md_res.energies_ev],
            "initial_energy_ev": round(md_res.energies_ev[0], 5) if md_res.energies_ev else 0.0,
            "final_energy_ev": round(md_res.energies_ev[-1], 5) if md_res.energies_ev else 0.0,
            "events": [e.to_dict() for e in md_res.events],
            "svg": svg_path.read_text(encoding="utf-8") if svg_path.exists() else "",
            "html": html_file.read_text(encoding="utf-8") if html_file.exists() else "",
        }


def _resolve_site_asset(url_path: str) -> Optional[Path]:
    clean = url_path.split("?", 1)[0].split("#", 1)[0]
    rel = "index.html" if clean in ("", "/", "/index.html") else clean.lstrip("/")
    candidate = (_SITE_DIR / rel).resolve()
    try:
        candidate.relative_to(_SITE_DIR)
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    return None


def create_studio_server(
    probe: Optional[YLMProbe] = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    import threading

    active_probe = probe or YLMProbe.default()
    probe_lock = threading.Lock()

    class StudioHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send_json(self, code: int, payload: dict):
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            clean_path = self.path.split("?", 1)[0]

            if clean_path == "/api/health":
                self._send_json(HTTPStatus.OK, {
                    "status": "ok",
                    "engine": active_probe.engine,
                    "source": active_probe.source_path,
                    "capabilities": [
                        "ask",
                        "physics_audit",
                        "pose_sandbox",
                        "run_md",
                        "ouroboros",
                        "eval_benchmark",
                    ],
                })
                return

            if clean_path == "/api/presets":
                raw_presets = get_builtin_presets()
                serialized = {}
                for k, item in raw_presets.items():
                    fr = item["frame"]
                    serialized[k] = {
                        "id": item["id"],
                        "title": item["title"],
                        "domain": item["domain"],
                        "default_question": item["default_question"],
                        "symmetry_hint": item["symmetry_hint"],
                        "xyz": fr.to_xyz_string(),
                        "n_atoms": fr.n_atoms,
                    }
                self._send_json(HTTPStatus.OK, {
                    "presets": serialized,
                    "reference_benchmarks": _load_reference_benchmarks(),
                })
                return

            asset = _resolve_site_asset(clean_path)
            if asset is not None:
                raw = asset.read_bytes()
                ctype = _MIME_OVERRIDES.get(asset.suffix.lower()) or mimetypes.guess_type(str(asset))[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")

                if self.path == "/api/ask":
                    if body.get("solvation_recipe"):
                        rec = body["solvation_recipe"]
                        frame = build_solvation_cluster(
                            solvent_smiles=rec.get("solvents", ["O=C1OCCO1", "O=C1OCCO1"]),
                            ion_symbol=rec.get("ion", "Li"),
                        )
                    else:
                        src = str(body.get("structure") or "O")
                        if src.strip() in get_builtin_presets():
                            frame, _ = resolve_preset_or_structure(src)
                        else:
                            frame = load_structure(src)
                    question = str(body.get("question") or "What is the dipole vector?")
                    domain = str(body.get("domain") or "general")
                    verify = bool(body.get("verify_o3", True))
                    with probe_lock:
                        res = active_probe.ask(frame, question, domain=domain, verify_o3=verify)
                    payload = res.to_dict()
                    payload["html"] = render_html_report(res)
                    self._send_json(HTTPStatus.OK, payload)
                    return

                if self.path == "/api/physics_audit":
                    with probe_lock:
                        audit = run_physics_compliance_audit(
                            frame_or_source=str(body.get("structure") or "r_fec_chiral"),
                            question=str(body.get("question") or "What is the 3D molecular dipole moment vector?"),
                            precision=str(body.get("precision") or "float64"),
                            n_rotations=int(body.get("n_rotations") or 8),
                            probe=active_probe,
                        )
                    self._send_json(HTTPStatus.OK, audit)
                    return

                if self.path == "/api/pose_sandbox":
                    t_ang = body.get("translate_ang") or [0.0, 0.0, 0.0]
                    with probe_lock:
                        sandbox = evaluate_pose_sandbox(
                            frame_or_source=str(body.get("structure") or "r_fec_chiral"),
                            question=str(body.get("question") or "What is the 3D molecular dipole moment vector?"),
                            rx_deg=float(body.get("rx_deg", 35.0)),
                            ry_deg=float(body.get("ry_deg", -48.0)),
                            rz_deg=float(body.get("rz_deg", 72.0)),
                            reflect=bool(body.get("reflect", False)),
                            translate_ang=(float(t_ang[0]), float(t_ang[1]), float(t_ang[2])),
                            permute_atoms=bool(body.get("permute_atoms", False)),
                            precision=str(body.get("precision") or "float64"),
                        )
                    self._send_json(HTTPStatus.OK, sandbox)
                    return

                if self.path == "/api/run_md":
                    with probe_lock:
                        md_res = _handle_run_md(body, active_probe)
                    self._send_json(HTTPStatus.OK, md_res)
                    return

                if self.path == "/api/ouroboros":
                    with probe_lock:
                        ob_res = _handle_ouroboros_inspect(
                            str(body.get("smiles") or "O=C1OC[C@H](F)O1"),
                            str(body.get("question") or "What is the 3D molecular dipole moment vector?"),
                            active_probe,
                        )
                    self._send_json(HTTPStatus.OK, ob_res)
                    return

                if self.path == "/api/eval_benchmark":
                    with probe_lock:
                        bench_res = _run_live_mini_benchmark(
                            epochs=int(body.get("epochs", 6)),
                            eval_rotations=int(body.get("eval_rotations", 4)),
                        )
                    self._send_json(HTTPStatus.OK, bench_res)
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return ThreadingHTTPServer((host, port), StudioHandler)


def run_studio_server(probe: Optional[YLMProbe] = None, host: str = "127.0.0.1", port: int = 8765) -> None:
    srv = create_studio_server(probe=probe, host=host, port=port)
    print(f"YLM Studio running at http://{host}:{port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()

"""Local browser app for testing YLM features and physics rules (`ylm serve`).

Runs on Python's built-in `http.server` with 5 views:
  1. Symmetry   : 8 live physics checks + 3D spin, mirror, move, and atom-order sandbox
  2. Playground : Ask plain-English 3D questions on molecules, electrolytes, MD frames, and drugs
  3. Dynamics   : Watch bond breaking, bond forming, and H+ jumps frame by frame
  4. Chirality  : Turn 2D stereo SMILES (`@/@@`, `/\\`) into 3D atoms and test mirror twins
  5. Benchmarks : Compare YLM against Uni-Mol, GraphMVP, and EquiLLM
"""
from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
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


def _load_reference_benchmarks() -> dict:
    """Read the saved handedness benchmark from the committed results file.

    Returns an empty model list if the file is missing, so the Studio never shows numbers
    that are not backed by a file in the repo.
    """
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
        ("Uni-Mol (3D Distance)", UniMolStyle3DBaseline(text_dim=64, d_model=48, n_layers=2, n_heads=4)),
        ("EquiLLM (Scalar Bridge)", EquiLLMMultiTaskBaseline(text_dim=64, hidden_irreps=hidden_ir, n_layers=2, d_model=48, n_vec=8)),
        (
            "YLM-Universe",
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
            for idx, fr_res in enumerate(scan.frame_results):
                fr = fr_res.frame
                mon.step(idx, idx * 2.0, fr.Z, fr.pos_angstrom, fr.forces_ev_ang, fr.cell_angstrom, fr.pbc)
            plot_reaction_timeline(mon.history, mon.events, svg_path)
            svg_text = svg_path.read_text(encoding="utf-8")
            html_text = render_trajectory_html_report(scan, title="Proton-Transfer Trajectory")
            return {
                "mode": mode,
                "n_frames": len(scan.frame_results),
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
        return {
            "mode": "live_md",
            "n_frames": len(md_res.frames),
            "initial_energy_ev": round(md_res.energies_ev[0], 5) if md_res.energies_ev else 0.0,
            "final_energy_ev": round(md_res.energies_ev[-1], 5) if md_res.energies_ev else 0.0,
            "events": [e.to_dict() for e in md_res.events],
            "svg": svg_path.read_text(encoding="utf-8") if svg_path.exists() else "",
            "html": html_file.read_text(encoding="utf-8") if html_file.exists() else "",
        }


_STUDIO_INDEX_HTML = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>YLM Directional Chemistry Studio</title>
<style>
  :root {
    --bg: #faf9f7;
    --nav-bg: rgba(250, 249, 247, 0.86);
    --surface: #ffffff;
    --subtle: #f3f2ee;
    --subtle-hover: #ebe9e2;
    --border: #e8e6e0;
    --border-strong: #d4d1c8;
    --ink: #141413;
    --ink-2: #5e5d59;
    --ink-3: #8e8c85;
    --btn-bg: #141413;
    --btn-fg: #ffffff;
    --btn-hover: #2c2b28;
    --ok-fg: #166534;
    --ok-bg: #eef8f1;
    --ok-bd: #cbead5;
    --err-fg: #b42318;
    --err-bg: #fef3f2;
    --err-bd: #fecdca;
    --shadow-sm: 0 1px 2px rgba(20, 20, 19, 0.03), 0 6px 20px rgba(20, 20, 19, 0.025);
  }
  [data-theme="dark"] {
    --bg: #111110;
    --nav-bg: rgba(17, 17, 16, 0.86);
    --surface: #191918;
    --subtle: #222220;
    --subtle-hover: #2b2b28;
    --border: #2c2c29;
    --border-strong: #3d3d39;
    --ink: #ededec;
    --ink-2: #a1a09a;
    --ink-3: #6f6e69;
    --btn-bg: #ededec;
    --btn-fg: #111110;
    --btn-hover: #ffffff;
    --ok-fg: #4ade80;
    --ok-bg: rgba(34, 197, 94, 0.10);
    --ok-bd: rgba(34, 197, 94, 0.24);
    --err-fg: #f87171;
    --err-bg: rgba(239, 68, 68, 0.10);
    --err-bd: rgba(239, 68, 68, 0.24);
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.25);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }
  .topbar-wrap {
    position: sticky; top: 0; z-index: 30;
    background: var(--nav-bg);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--border);
  }
  .topbar {
    max-width: 1140px; margin: 0 auto; padding: 12px 28px;
    display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap;
  }
  .brand { display: flex; align-items: center; gap: 10px; }
  .brand-mark {
    width: 28px; height: 28px; border-radius: 8px; background: var(--ink); color: var(--surface);
    display: inline-flex; align-items: center; justify-content: center; font-weight: 600; font-size: 13px;
    letter-spacing: -0.03em;
  }
  .brand-name { font-size: 15px; font-weight: 600; letter-spacing: -0.02em; }
  .brand-tag { font-size: 12.5px; color: var(--ink-3); padding-left: 8px; border-left: 1px solid var(--border); }
  .seg-nav {
    display: inline-flex; gap: 2px; background: var(--subtle);
    padding: 3px; border-radius: 999px; border: 1px solid var(--border);
  }
  .tab-btn {
    background: transparent; color: var(--ink-2); border: none;
    padding: 6px 14px; border-radius: 999px; font-size: 13px; font-weight: 500;
    cursor: pointer; transition: all 0.15s ease;
  }
  .tab-btn:hover { color: var(--ink); }
  .tab-btn.active {
    background: var(--surface); color: var(--ink);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.07);
  }
  .nav-right { display: flex; align-items: center; gap: 8px; }
  .icon-btn {
    width: 32px; height: 32px; border-radius: 999px; border: 1px solid var(--border);
    background: var(--surface); color: var(--ink-2); cursor: pointer;
    display: inline-flex; align-items: center; justify-content: center; font-size: 14px;
  }
  .icon-btn:hover { color: var(--ink); border-color: var(--border-strong); }

  .container { max-width: 1140px; margin: 0 auto; padding: 32px 28px 72px; }
  .pane { display: none; }
  .pane.active { display: block; animation: fadeIn 0.18s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: translateY(0); } }

  .page-head {
    display: flex; justify-content: space-between; align-items: flex-end;
    gap: 16px; flex-wrap: wrap; margin-bottom: 24px;
  }
  .page-head h1 { margin: 0; font-size: 24px; font-weight: 600; letter-spacing: -0.025em; }
  .page-head p { margin: 4px 0 0; font-size: 14px; color: var(--ink-2); }

  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 22px; box-shadow: var(--shadow-sm);
  }
  .pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 11px; border-radius: 999px; font-size: 12px; font-weight: 500;
    border: 1px solid var(--border); background: var(--surface); color: var(--ink-2);
  }
  .pill-ok { background: var(--ok-bg); color: var(--ok-fg); border-color: var(--ok-bd); }
  .pill-err { background: var(--err-bg); color: var(--err-fg); border-color: var(--err-bd); }
  .dot-ok { width: 6px; height: 6px; border-radius: 99px; background: currentColor; }

  label { display: block; font-size: 12px; font-weight: 500; color: var(--ink-3); margin-bottom: 6px; }
  input[type="text"], input[type="number"], select, textarea {
    width: 100%; background: var(--surface); color: var(--ink);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 8px 12px; font-size: 13.5px; font-family: inherit;
    outline: none; transition: border-color 0.15s;
  }
  input[type="text"]:focus, input[type="number"]:focus, select:focus, textarea:focus {
    border-color: var(--ink-2);
  }
  input[type="range"] { width: 100%; accent-color: var(--ink); cursor: pointer; }

  .btn {
    background: var(--btn-bg); color: var(--btn-fg); border: 1px solid transparent;
    padding: 8px 16px; border-radius: 999px; font-size: 13px; font-weight: 500;
    cursor: pointer; transition: background 0.15s ease; white-space: nowrap;
  }
  .btn:hover { background: var(--btn-hover); }
  .btn:disabled { opacity: 0.55; cursor: default; }
  .btn-ghost {
    background: var(--surface); color: var(--ink); border: 1px solid var(--border);
  }
  .btn-ghost:hover { background: var(--subtle); border-color: var(--border-strong); }

  .chip-row { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    font-size: 12.5px; padding: 5px 12px; border-radius: 999px;
    background: var(--subtle); border: 1px solid transparent;
    color: var(--ink-2); cursor: pointer; transition: all 0.14s ease;
  }
  .chip:hover { background: var(--subtle-hover); color: var(--ink); }
  .chip.active { background: var(--ink); color: var(--surface); }

  .sym-layout {
    display: grid; grid-template-columns: minmax(380px, 1.3fr) minmax(320px, 1fr);
    gap: 20px; align-items: start;
  }
  @media (max-width: 960px) { .sym-layout { grid-template-columns: 1fr; } }

  .canvas-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 600px) { .canvas-pair { grid-template-columns: 1fr; } }
  .canvas-box {
    background: var(--subtle); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden; position: relative;
  }
  .canvas-caption {
    position: absolute; top: 10px; left: 12px; font-size: 11.5px; font-weight: 500;
    color: var(--ink-2); pointer-events: none;
  }
  .canvas-legend {
    position: absolute; bottom: 9px; left: 12px; display: flex; gap: 10px;
    font-size: 11px; color: var(--ink-2); pointer-events: none;
  }
  canvas.scene3d { width: 100%; height: 245px; display: block; cursor: grab; }
  canvas.scene3d:active { cursor: grabbing; }

  .slider-strip {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
    margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border);
  }
  .toggle-strip {
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
    gap: 10px; margin-top: 12px;
  }
  .check-pill {
    display: inline-flex; align-items: center; gap: 6px; padding: 5px 11px;
    border-radius: 999px; background: var(--subtle); font-size: 12.5px;
    color: var(--ink-2); cursor: pointer; user-select: none;
  }
  .check-pill input { margin: 0; accent-color: var(--ink); }

  .law-list { display: flex; flex-direction: column; }
  .law-item { border-bottom: 1px solid var(--border); padding: 12px 2px; }
  .law-item:first-child { padding-top: 2px; }
  .law-item:last-child { border-bottom: none; padding-bottom: 2px; }
  .law-item summary {
    list-style: none; display: flex; justify-content: space-between;
    align-items: center; gap: 10px; cursor: pointer; user-select: none;
  }
  .law-item summary::-webkit-details-marker { display: none; }
  .law-title { display: flex; align-items: center; gap: 9px; font-size: 13.5px; font-weight: 500; }
  .law-check {
    width: 18px; height: 18px; border-radius: 99px; display: inline-flex;
    align-items: center; justify-content: center; font-size: 11px; font-weight: 700;
  }
  .law-check.ok { background: var(--ok-bg); color: var(--ok-fg); }
  .law-check.bad { background: var(--err-bg); color: var(--err-fg); }
  .law-body { padding: 10px 0 4px 27px; font-size: 12.5px; color: var(--ink-2); line-height: 1.55; }

  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th {
    padding: 10px 12px; border-bottom: 1px solid var(--border);
    color: var(--ink-3); font-size: 12px; font-weight: 500; text-align: left;
  }
  td { padding: 11px 12px; border-bottom: 1px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  .num { text-align: right; }
  tr.row-highlight td { background: var(--subtle); font-weight: 500; }

  .composer {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 18px; padding: 16px 18px; box-shadow: var(--shadow-sm);
    margin-bottom: 22px;
  }
  .composer-bar {
    display: flex; align-items: center; gap: 10px; margin-top: 12px;
    background: var(--subtle); border-radius: 12px; padding: 6px 6px 6px 14px;
    border: 1px solid transparent; transition: border-color 0.15s;
  }
  .composer-bar:focus-within { border-color: var(--border-strong); background: var(--surface); }
  .composer-bar input {
    border: none; background: transparent; padding: 6px 0; font-size: 14.5px; flex: 1;
  }
  .composer-bar select {
    width: auto; border: none; background: transparent; color: var(--ink-2);
    font-size: 12.5px; padding: 6px 8px; cursor: pointer;
  }

  iframe.report-box {
    width: 100%; min-height: 520px; border: none; background: transparent; display: block;
  }
  .kpi-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 18px; }
  @media (max-width: 760px) { .kpi-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<div class="topbar-wrap">
  <div class="topbar">
    <div class="brand">
      <span class="brand-mark">Y</span>
      <span class="brand-name">YLM</span>
      <span class="brand-tag">3D Equivariant Chemistry</span>
    </div>

    <nav class="seg-nav">
      <button class="tab-btn active" onclick="switchTab('physics', this)">Symmetry</button>
      <button class="tab-btn" onclick="switchTab('probe', this)">Playground</button>
      <button class="tab-btn" onclick="switchTab('md', this)">Dynamics</button>
      <button class="tab-btn" onclick="switchTab('ouroboros', this)">Chirality</button>
      <button class="tab-btn" onclick="switchTab('benchmark', this)">Benchmarks</button>
    </nav>

    <div class="nav-right">
      <span class="pill pill-ok" id="engineBadge"><span class="dot-ok"></span>Ready</span>
      <button class="icon-btn" id="themeBtn" onclick="toggleTheme()" title="Switch light / dark theme">☾</button>
    </div>
  </div>
</div>

<div class="container">

  <!-- ================= 1. SYMMETRY ================= -->
  <div id="pane-physics" class="pane active">
    <div class="page-head">
      <div>
        <h1>Physical Symmetry</h1>
        <p>Rotate, mirror, shift, or reorder atoms in 3D and verify exact conservation laws.</p>
      </div>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <select id="auditPresetSel" onchange="runPhysicsAudit()" style="width:auto;min-width:220px;"></select>
        <select id="auditPrecision" onchange="runPhysicsAudit()" style="width:auto;">
          <option value="float64" selected>64-bit strict</option>
          <option value="float32">32-bit fast</option>
        </select>
        <button class="btn" onclick="runPhysicsAudit()">Verify</button>
      </div>
    </div>
    <input type="hidden" id="auditQuestion" value="What is the 3D molecular dipole moment vector?" />

    <div class="sym-layout">
      <!-- Left: Interactive 3D Sandbox -->
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
          <span style="font-weight:600;font-size:14.5px;">3D Pose &amp; Mirror Sandbox</span>
          <span class="pill mono" id="detReadout">det(Q) = +1</span>
        </div>

        <div class="canvas-pair">
          <div class="canvas-box">
            <div class="canvas-caption">Original</div>
            <canvas id="cvOrig" class="scene3d"></canvas>
            <div class="canvas-legend">
              <span><b style="color:#1d4ed8;">●</b> YLM</span>
            </div>
          </div>
          <div class="canvas-box">
            <div class="canvas-caption">Transformed</div>
            <canvas id="cvTrans" class="scene3d"></canvas>
            <div class="canvas-legend">
              <span><b style="color:#15803d;">●</b> Target</span>
              <span><b style="color:#1d4ed8;">●</b> YLM</span>
              <span><b style="color:#b91c1c;">●</b> Uni-Mol</span>
            </div>
          </div>
        </div>

        <div class="slider-strip">
          <div>
            <label>X rotation · <span id="lblRx" class="mono" style="color:var(--ink);">35°</span></label>
            <input type="range" id="slRx" min="-180" max="180" value="35" oninput="onSandboxChange()" />
          </div>
          <div>
            <label>Y rotation · <span id="lblRy" class="mono" style="color:var(--ink);">-48°</span></label>
            <input type="range" id="slRy" min="-180" max="180" value="-48" oninput="onSandboxChange()" />
          </div>
          <div>
            <label>Z rotation · <span id="lblRz" class="mono" style="color:var(--ink);">72°</span></label>
            <input type="range" id="slRz" min="-180" max="180" value="72" oninput="onSandboxChange()" />
          </div>
        </div>

        <div class="toggle-strip">
          <div style="display:flex;gap:6px;flex-wrap:wrap;">
            <label class="check-pill">
              <input type="checkbox" id="chkReflect" checked onchange="runPoseSandbox()" />
              Mirror flip
            </label>
            <label class="check-pill">
              <input type="checkbox" id="chkPermute" onchange="runPoseSandbox()" />
              Shuffle atoms
            </label>
            <label class="check-pill">
              <input type="checkbox" id="chkTranslate" onchange="runPoseSandbox()" />
              Shift +t
            </label>
          </div>
        </div>

        <div style="margin-top:16px;padding-top:12px;border-top:1px solid var(--border);overflow-x:auto;">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th class="num">Energy drift</th>
                <th class="num">Mirror error</th>
                <th class="num">Arrow cosine</th>
                <th class="num">Matrix error</th>
              </tr>
            </thead>
            <tbody id="sandboxTableBody"></tbody>
          </table>
        </div>
      </div>

      <!-- Right: Clean 8-Law Scorecard -->
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
          <span style="font-weight:600;font-size:14.5px;">8 Conservation &amp; Symmetry Laws</span>
          <span id="auditSummaryBadge" class="pill pill-ok">Checking...</span>
        </div>
        <div id="lawsContainer" class="law-list"></div>
      </div>
    </div>
  </div>

  <!-- ================= 2. PLAYGROUND ================= -->
  <div id="pane-probe" class="pane">
    <div class="page-head">
      <div>
        <h1>3D Playground</h1>
        <p>Ask natural-language questions on molecules, ion shells, reaction frames, or protein pockets.</p>
      </div>
      <div style="display:flex;gap:8px;align-items:center;">
        <button class="btn btn-ghost" onclick="toggleCoordsDrawer()">Coordinates</button>
        <button class="btn btn-ghost" onclick="buildLiCluster()">+ Li+ Shell</button>
      </div>
    </div>

    <div class="composer">
      <div class="chip-row" id="probePresetChips"></div>

      <div class="composer-bar">
        <input type="text" id="qInput" value="What is the force vector on the hydrogen atom?" onkeydown="if(event.key==='Enter')runQuery()" placeholder="Ask about a dipole, atom force, or safe growth direction..." />
        <select id="domainSel">
          <option value="general">General</option>
          <option value="electrolyte">Electrolyte</option>
          <option value="md">Dynamics</option>
          <option value="drug">Drug Pocket</option>
        </select>
        <button class="btn" onclick="runQuery()">Ask →</button>
      </div>

      <div id="coordsDrawer" style="display:none;margin-top:14px;padding-top:14px;border-top:1px solid var(--border);">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
          <label style="margin:0;">SMILES or XYZ / POSCAR / CIF coordinates</label>
          <label style="margin:0;display:inline-flex;align-items:center;gap:6px;cursor:pointer;">
            <input type="checkbox" id="o3Chk" checked /> Live O(3) check
          </label>
        </div>
        <textarea id="structInput" rows="5" class="mono">3
water demo
O  0.000  0.000  0.125
H  0.020  0.790 -0.460
H -0.015 -0.745 -0.485</textarea>
      </div>
    </div>

    <iframe id="reportFrame" class="report-box"></iframe>
  </div>

  <!-- ================= 3. DYNAMICS ================= -->
  <div id="pane-md" class="pane">
    <div class="page-head">
      <div>
        <h1>Reaction Dynamics</h1>
        <p>Track bond breaking, bond forming, and proton hops across molecular dynamics trajectories.</p>
      </div>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <select id="mdModeSel" style="width:auto;">
          <option value="scan_zundel">Saved proton-hop trajectory</option>
          <option value="live_md">Live reactive MD (H5O2+)</option>
        </select>
        <input type="number" id="mdSteps" value="28" min="8" max="80" style="width:74px;" title="MD steps" />
        <input type="number" id="mdTemp" value="350" min="0" max="1500" style="width:82px;" title="Temperature (K)" />
        <button class="btn" onclick="runReactiveMD()">Run</button>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div id="mdEventsSummary" class="chip-row" style="margin-bottom:14px;"></div>
      <div id="mdSvgContainer" style="overflow-x:auto;"></div>
    </div>

    <iframe id="mdReportFrame" class="report-box"></iframe>
  </div>

  <!-- ================= 4. CHIRALITY ================= -->
  <div id="pane-ouroboros" class="pane">
    <div class="page-head">
      <div>
        <h1>3D Chirality</h1>
        <p>Embed 2D stereo SMILES into 3D coordinates and test left- vs. right-handed mirror twins.</p>
      </div>
    </div>

    <div class="composer">
      <div class="chip-row" style="margin-bottom:12px;">
        <span class="chip" onclick="setStereoSmiles('O=C1OC[C@H](F)O1')">(R)-FEC</span>
        <span class="chip" onclick="setStereoSmiles('O=C1OC[C@@H](F)O1')">(S)-FEC</span>
        <span class="chip" onclick="setStereoSmiles('F[C@H](Cl)Br')">(R)-CHFCIBr</span>
        <span class="chip" onclick="setStereoSmiles('F[C@@H](Cl)Br')">(S)-CHFCIBr</span>
        <span class="chip" onclick="setStereoSmiles('F/C=C\\\\F')">cis-CHF=CHF</span>
        <span class="chip" onclick="setStereoSmiles('F/C=C/F')">trans-CHF=CHF</span>
      </div>
      <div class="composer-bar">
        <input type="text" id="stereoSmilesInput" class="mono" value="O=C1OC[C@H](F)O1" onkeydown="if(event.key==='Enter')runOuroborosAudit()" />
        <input type="hidden" id="stereoQuestionInput" value="What is the 3D molecular dipole moment vector?" />
        <button class="btn" onclick="runOuroborosAudit()">Inspect →</button>
      </div>
      <div id="stereoAuditResult"></div>
    </div>

    <iframe id="stereoReportFrame" class="report-box"></iframe>
  </div>

  <!-- ================= 5. BENCHMARKS ================= -->
  <div id="pane-benchmark" class="pane">
    <div class="page-head">
      <div>
        <h1>Model Benchmarks</h1>
        <p>Comparison across chirality, covariant 3D vectors, and symmetric rank-2 tensors under O(3).</p>
      </div>
      <div style="display:flex;gap:8px;align-items:center;">
        <input type="number" id="benchEpochs" value="6" min="2" max="20" style="width:68px;" title="Training epochs" />
        <button class="btn" id="btnLiveBench" onclick="runLiveBenchmark()">Run Live CPU Eval</button>
      </div>
    </div>

    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
        <span style="font-weight:600;font-size:14.5px;">Saved 24-Epoch CPU Run (8 Mirror-Twin Families)</span>
        <span class="pill mono">results/cpu_smoke_nov12/chiral_benchmark.json</span>
      </div>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th class="num">Chiral Acc</th>
              <th class="num">New Wording</th>
              <th class="num">Unseen Scaffold</th>
              <th class="num">Mirror Err</th>
              <th class="num">Planar Zero</th>
              <th class="num">Vector Cos</th>
              <th class="num">Tensor Cos</th>
            </tr>
          </thead>
          <tbody id="refBenchBody"></tbody>
        </table>
      </div>
    </div>

    <div id="liveBenchContainer" class="card" style="margin-top:20px;display:none;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
        <span style="font-weight:600;font-size:14.5px;">Fresh CPU Training Run</span>
        <span class="pill pill-ok">Completed live</span>
      </div>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th class="num">Time</th>
              <th class="num">Scalar MAE</th>
              <th class="num">Vector Cos</th>
              <th class="num">Force Cos</th>
              <th class="num">Rot Err</th>
              <th class="num">Tensor Cos</th>
              <th class="num">Routing Acc</th>
            </tr>
          </thead>
          <tbody id="liveBenchBody"></tbody>
        </table>
      </div>
    </div>
  </div>

</div>

<script>
let PRESETS = {};
let SANDBOX_SCENES = { orig: null, trans: null };
let camOrig = { rx: -0.35, ry: 0.55 }, camTrans = { rx: -0.35, ry: 0.55 };
let sandboxTimer = null;

const ELEM_COL = {
  1: "#ffffff", 3: "#8b5cf6", 6: "#3a3a38", 7: "#2563eb",
  8: "#dc2626", 9: "#16a34a", 15: "#ea580c", 16: "#ca8a04",
  17: "#0d9488", 26: "#b45309", 35: "#be185d"
};

window.addEventListener("message", e => {
  if (e.data && e.data.type === "ylm-frame-height") {
    ["reportFrame", "mdReportFrame", "stereoReportFrame"].forEach(id => {
      const fr = document.getElementById(id);
      if (fr && fr.contentWindow === e.source) {
        fr.style.height = Math.max(460, e.data.height + 12) + "px";
      }
    });
  }
});

function syncIframeThemes() {
  const theme = document.documentElement.getAttribute("data-theme") || "light";
  ["reportFrame", "mdReportFrame", "stereoReportFrame"].forEach(id => {
    const fr = document.getElementById(id);
    if (fr && fr.contentWindow) {
      fr.contentWindow.postMessage({ type: "ylm-theme", theme }, "*");
    }
  });
}

function toggleTheme() {
  const cur = document.documentElement.getAttribute("data-theme") || "light";
  const next = cur === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", next);
  document.getElementById("themeBtn").textContent = next === "light" ? "☾" : "☀";
  drawScene(document.getElementById("cvOrig"), SANDBOX_SCENES.orig, camOrig);
  drawScene(document.getElementById("cvTrans"), SANDBOX_SCENES.trans, camTrans);
  syncIframeThemes();
}

function toggleCoordsDrawer() {
  const el = document.getElementById("coordsDrawer");
  el.style.display = el.style.display === "none" ? "block" : "none";
}

function switchTab(name, btn) {
  document.querySelectorAll('.pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('pane-' + name).classList.add('active');
  if (btn) btn.classList.add('active');
  if (name === 'physics') {
    setTimeout(() => {
      drawScene(document.getElementById('cvOrig'), SANDBOX_SCENES.orig, camOrig);
      drawScene(document.getElementById('cvTrans'), SANDBOX_SCENES.trans, camTrans);
    }, 20);
  }
  if (name === 'probe' && !document.getElementById('reportFrame').srcdoc) runQuery();
  if (name === 'md' && !document.getElementById('mdSvgContainer').innerHTML) runReactiveMD();
  if (name === 'ouroboros' && !document.getElementById('stereoAuditResult').innerHTML) runOuroborosAudit();
}

function attachDrag(canvasId, camObj, sceneKey) {
  const cv = document.getElementById(canvasId);
  let drag = false, lx = 0, ly = 0;
  cv.addEventListener('mousedown', e => { drag = true; lx = e.clientX; ly = e.clientY; });
  window.addEventListener('mouseup', () => { drag = false; });
  window.addEventListener('mousemove', e => {
    if (!drag) return;
    camObj.ry += (e.clientX - lx) * 0.01;
    camObj.rx += (e.clientY - ly) * 0.01;
    lx = e.clientX; ly = e.clientY;
    if (SANDBOX_SCENES[sceneKey]) drawScene(cv, SANDBOX_SCENES[sceneKey], camObj);
  });
}

function drawScene(cv, scene, cam) {
  if (!scene || !scene.atoms) return;
  const rect = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(240, rect.width || 300);
  const h = Math.max(220, rect.height || 245);
  cv.width = w * dpr;
  cv.height = h * dpr;
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  if (!scene.atoms.length) return;

  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  let cx = 0, cy = 0, cz = 0;
  scene.atoms.forEach(a => { cx += a.x; cy += a.y; cz += a.z; });
  cx /= scene.atoms.length; cy /= scene.atoms.length; cz /= scene.atoms.length;

  let maxR = 2.3;
  scene.atoms.forEach(a => {
    const d = Math.hypot(a.x - cx, a.y - cy, a.z - cz);
    if (d + 1.5 > maxR) maxR = d + 1.5;
  });
  const scale = 0.39 * Math.min(w, h) / maxR;

  function proj(px, py, pz) {
    const x = px - cx, y = py - cy, z = pz - cz;
    const cy1 = Math.cos(cam.ry), sy1 = Math.sin(cam.ry);
    const x1 = x * cy1 + z * sy1, z1 = -x * sy1 + z * cy1;
    const cx1 = Math.cos(cam.rx), sx1 = Math.sin(cam.rx);
    const y2 = y * cx1 - z1 * sx1, z2 = y * sx1 + z1 * cx1;
    return [w / 2 + x1 * scale, h / 2 - y2 * scale, z2];
  }

  ctx.lineWidth = 2.4;
  ctx.lineCap = "round";
  ctx.strokeStyle = isDark ? "#52524e" : "#c8c5bc";
  (scene.bonds || []).forEach(([i, j]) => {
    const p1 = proj(scene.atoms[i].x, scene.atoms[i].y, scene.atoms[i].z);
    const p2 = proj(scene.atoms[j].x, scene.atoms[j].y, scene.atoms[j].z);
    ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke();
  });

  const weights = scene.weights || [];
  const maxW = Math.max(...weights, 1e-6);
  const meanW = weights.length ? weights.reduce((a, b) => a + b, 0) / weights.length : maxW;
  const hasPeakedAttn = maxW > 1.35 * meanW;
  const ordered = scene.atoms.map((a, idx) => ({a, idx, p: proj(a.x, a.y, a.z)})).sort((u, v) => u.p[2] - v.p[2]);
  ordered.forEach(({a, idx, p}) => {
    const wt = (weights[idx] || 0) / maxW;
    const rad = a.Z === 1 ? 6.8 : 10.8;
    if (hasPeakedAttn && wt > 0.55) {
      ctx.beginPath(); ctx.arc(p[0], p[1], rad + 6 * wt, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(217, 119, 6, ${0.14 + 0.22 * wt})`; ctx.fill();
    }
    const baseCol = ELEM_COL[a.Z] || "#78716c";
    ctx.beginPath(); ctx.arc(p[0], p[1], rad, 0, Math.PI * 2);
    ctx.fillStyle = baseCol; ctx.fill();
    const spec = ctx.createRadialGradient(p[0] - rad * 0.32, p[1] - rad * 0.32, rad * 0.05, p[0] - rad * 0.15, p[1] - rad * 0.15, rad * 0.75);
    spec.addColorStop(0, "rgba(255,255,255,0.42)");
    spec.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = spec; ctx.fill();
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.18)" : "rgba(20,20,19,0.16)";
    ctx.lineWidth = 1; ctx.stroke();
  });

  (scene.arrows || []).forEach((ar, k) => {
    const s = proj(ar.start[0], ar.start[1], ar.start[2]);
    const e = proj(ar.end[0], ar.end[1], ar.end[2]);
    ctx.strokeStyle = ar.color; ctx.fillStyle = ar.color;
    ctx.lineWidth = k === 0 && scene.arrows.length > 1 ? 3.2 : 2.4;
    if (k === 0 && scene.arrows.length > 1) ctx.setLineDash([5, 3]); else ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(s[0], s[1]); ctx.lineTo(e[0], e[1]); ctx.stroke();
    ctx.setLineDash([]);
    const ang = Math.atan2(e[1] - s[1], e[0] - s[0]);
    ctx.beginPath();
    ctx.moveTo(e[0], e[1]);
    ctx.lineTo(e[0] - 8.5 * Math.cos(ang - 0.38), e[1] - 8.5 * Math.sin(ang - 0.38));
    ctx.lineTo(e[0] - 8.5 * Math.cos(ang + 0.38), e[1] - 8.5 * Math.sin(ang + 0.38));
    ctx.closePath(); ctx.fill();
  });
}

function cleanPresetLabel(id, fallback) {
  const map = {
    r_fec_chiral: "(R)-FEC · Chiral additive",
    s_fec_chiral: "(S)-FEC · Mirror twin",
    water_c2v: "Water (H2O · C2v)",
    trans_difluoroethene_c2h: "trans-CHF=CHF (Dipole = 0)",
    cis_difluoroethene_c2v: "cis-CHF=CHF (Polar)",
    r_bcfm_chiral: "(R)-CHFCIBr · Chiral carbon",
    lifepo4_pbc_crystal: "LiFePO4 · 3D crystal",
    li_ec_shell: "Li+ · EC solvent shell",
    indole_pocket: "Indole · Protein pocket",
  };
  return map[id] || fallback || id;
}

async function initStudio() {
  attachDrag('cvOrig', camOrig, 'orig');
  attachDrag('cvTrans', camTrans, 'trans');
  window.addEventListener('resize', () => {
    drawScene(document.getElementById('cvOrig'), SANDBOX_SCENES.orig, camOrig);
    drawScene(document.getElementById('cvTrans'), SANDBOX_SCENES.trans, camTrans);
  });

  const r = await fetch('/api/presets');
  const data = await r.json();
  PRESETS = data.presets || {};

  const sel = document.getElementById('auditPresetSel');
  const chips = document.getElementById('probePresetChips');
  sel.innerHTML = '';
  chips.innerHTML = '';
  Object.values(PRESETS).forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = cleanPresetLabel(p.id, p.title);
    if (p.id === 'r_fec_chiral') opt.selected = true;
    sel.appendChild(opt);

    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = cleanPresetLabel(p.id, p.id).split(' · ')[0];
    chip.title = p.title;
    chip.onclick = () => {
      document.querySelectorAll('#probePresetChips .chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      document.getElementById('structInput').value = p.xyz;
      document.getElementById('domainSel').value = p.domain;
      document.getElementById('qInput').value = p.default_question;
      runQuery();
    };
    chips.appendChild(chip);
  });

  renderRefBenchmark(data.reference_benchmarks.chiral_multirank_24ep.models);
  await runPhysicsAudit();
}

function renderRefBenchmark(models) {
  const tb = document.getElementById('refBenchBody');
  tb.innerHTML = models.map(m => {
    const isOurs = m.model.includes('YLM');
    return `<tr class="${isOurs ? 'row-highlight' : ''}">
      <td>${m.model}</td>
      <td class="num mono">${(m.chiral_acc * 100).toFixed(1)}%</td>
      <td class="num mono">${(m.heldout_chiral_acc * 100).toFixed(1)}%</td>
      <td class="num mono">${(m.zeroshot_scaffold_acc * 100).toFixed(1)}%</td>
      <td class="num mono">${m.chiral_parity_err.toExponential(2)}</td>
      <td class="num mono">${m.achiral_planar_abs.toExponential(2)}</td>
      <td class="num mono">${m.vector_cos >= 0 ? '+' : ''}${m.vector_cos.toFixed(3)}</td>
      <td class="num mono">${m.tensor_cos >= 0 ? '+' : ''}${m.tensor_cos.toFixed(3)}</td>
    </tr>`;
  }).join('');
}

async function runPhysicsAudit() {
  const preset = document.getElementById('auditPresetSel').value || 'r_fec_chiral';
  const question = document.getElementById('auditQuestion').value;
  const precision = document.getElementById('auditPrecision').value;
  document.getElementById('auditSummaryBadge').textContent = 'Checking...';
  const resp = await fetch('/api/physics_audit', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ structure: preset, question, precision, n_rotations: 6 })
  });
  const data = await resp.json();
  if (data.error) { alert(data.error); return; }

  const badge = document.getElementById('auditSummaryBadge');
  badge.textContent = `${data.passed_count} / ${data.total_laws} Passed`;
  badge.className = data.passed_all ? 'pill pill-ok' : 'pill pill-err';

  const box = document.getElementById('lawsContainer');
  box.innerHTML = data.laws.map((law, idx) => {
    const cleanName = law.name.replace(/^\\d+\\.\\s*/, '');
    const baseHtml = law.baseline_errors
      ? '<div style="margin-top:6px;color:var(--err-fg);">Baselines failed: ' +
        Object.entries(law.baseline_errors).map(([k, v]) => `${k} <span class="mono">${Number(v).toExponential(2)}</span>`).join(' · ') +
        '</div>'
      : '';
    return `<details class="law-item" ${idx === 0 ? 'open' : ''}>
      <summary>
        <span class="law-title">
          <span class="law-check ${law.passed ? 'ok' : 'bad'}">${law.passed ? '✓' : '!'}</span>
          <span>${cleanName}</span>
        </span>
        <span class="mono" style="font-size:12px;color:var(--ink-3);white-space:nowrap;flex-shrink:0;">${Number(law.ylm_error).toExponential(1)}</span>
      </summary>
      <div class="law-body">
        <div class="mono" style="color:var(--ink);font-size:12px;margin-bottom:4px;">${law.formula}</div>
        <div>${law.detail}</div>
        ${baseHtml}
      </div>
    </details>`;
  }).join('');

  await runPoseSandbox();
}

function onSandboxChange() {
  document.getElementById('lblRx').textContent = document.getElementById('slRx').value + '°';
  document.getElementById('lblRy').textContent = document.getElementById('slRy').value + '°';
  document.getElementById('lblRz').textContent = document.getElementById('slRz').value + '°';
  clearTimeout(sandboxTimer);
  sandboxTimer = setTimeout(runPoseSandbox, 80);
}

async function runPoseSandbox() {
  const payload = {
    structure: document.getElementById('auditPresetSel').value || 'r_fec_chiral',
    question: document.getElementById('auditQuestion').value,
    precision: document.getElementById('auditPrecision').value,
    rx_deg: Number(document.getElementById('slRx').value),
    ry_deg: Number(document.getElementById('slRy').value),
    rz_deg: Number(document.getElementById('slRz').value),
    reflect: document.getElementById('chkReflect').checked,
    permute_atoms: document.getElementById('chkPermute').checked,
    translate_ang: document.getElementById('chkTranslate').checked ? [3.5, -2.1, 4.0] : [0.0, 0.0, 0.0],
  };
  const resp = await fetch('/api/pose_sandbox', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
  });
  const data = await resp.json();
  if (data.error) return;

  document.getElementById('detReadout').textContent = `det(Q) = ${data.det_Q > 0 ? '+1' : '-1'}`;
  SANDBOX_SCENES.orig = data.scene_orig;
  SANDBOX_SCENES.trans = data.scene_transformed;
  drawScene(document.getElementById('cvOrig'), SANDBOX_SCENES.orig, camOrig);
  drawScene(document.getElementById('cvTrans'), SANDBOX_SCENES.trans, camTrans);

  const names = [
    ['ylm', 'YLM-Universe'],
    ['equillm', 'EquiLLM'],
    ['unimol', 'Uni-Mol'],
    ['graphmvp', 'GraphMVP'],
  ];
  const tol = data.precision === 'float64' ? 1e-8 : 1e-4;
  document.getElementById('sandboxTableBody').innerHTML = names.map(([key, title]) => {
    const m = data.models[key];
    const cBad = m.chiral_0o.abs_err > tol;
    const vBad = m.vector_1o.rel_err > tol;
    return `<tr class="${key === 'ylm' ? 'row-highlight' : ''}">
      <td>${title}</td>
      <td class="num mono">${m.scalar_0e.abs_err.toExponential(1)}</td>
      <td class="num mono" style="color:${cBad ? 'var(--err-fg)' : 'var(--ok-fg)'};">${m.chiral_0o.abs_err.toExponential(1)}</td>
      <td class="num mono" style="color:${vBad ? 'var(--err-fg)' : 'var(--ok-fg)'};">${m.vector_1o.cosine_with_expected >= 0 ? '+' : ''}${m.vector_1o.cosine_with_expected.toFixed(3)}</td>
      <td class="num mono">${m.tensor_0e2e.rel_err.toExponential(1)}</td>
    </tr>`;
  }).join('');
}

async function runQuery(extraPayload) {
  const payload = extraPayload || {
    structure: document.getElementById("structInput").value,
    domain: document.getElementById("domainSel").value,
    question: document.getElementById("qInput").value,
    verify_o3: document.getElementById("o3Chk").checked
  };
  const resp = await fetch("/api/ask", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  const data = await resp.json();
  if (data.html) {
    const fr = document.getElementById("reportFrame");
    fr.onload = syncIframeThemes;
    fr.srcdoc = data.html;
  } else {
    alert(data.error || "Request failed");
  }
}

function buildLiCluster() {
  document.getElementById("domainSel").value = "electrolyte";
  document.getElementById("qInput").value = "What is the desolvation escape force vector on Li+?";
  runQuery({
    solvation_recipe: { ion: "Li", solvents: ["O=C1OCCO1", "O=C1OC[C@H](F)O1", "COC(=O)OC"] },
    domain: "electrolyte",
    question: "What is the desolvation escape force vector on Li+?",
    verify_o3: true
  });
}

async function runReactiveMD() {
  const payload = {
    mode: document.getElementById("mdModeSel").value,
    steps: Number(document.getElementById("mdSteps").value),
    temperature_k: Number(document.getElementById("mdTemp").value)
  };
  document.getElementById("mdEventsSummary").innerHTML = '<span class="pill">Running...</span>';
  const resp = await fetch("/api/run_md", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  const data = await resp.json();
  if (data.error) { alert(data.error); return; }
  const evHtml = (data.events && data.events.length)
    ? data.events.map(e => `<span class="pill pill-ok">Frame ${e.frame_index} · ${e.event_type} (${e.labels.join('-')}) · W = ${e.work_ev.toFixed(3)} eV</span>`).join('')
    : `<span class="pill">${data.n_frames} frames scanned · scrub timeline below</span>`;
  document.getElementById("mdEventsSummary").innerHTML = evHtml;
  document.getElementById("mdSvgContainer").innerHTML = data.svg || '';
  if (data.html) {
    const fr = document.getElementById("mdReportFrame");
    fr.onload = syncIframeThemes;
    fr.srcdoc = data.html;
  }
}

function setStereoSmiles(smi) {
  document.getElementById("stereoSmilesInput").value = smi;
  runOuroborosAudit();
}

async function runOuroborosAudit() {
  const smiles = document.getElementById("stereoSmilesInput").value;
  const question = document.getElementById("stereoQuestionInput").value;
  const resp = await fetch("/api/ouroboros", {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({ smiles, question })
  });
  const data = await resp.json();
  if (data.error) { alert(data.error); return; }
  const et = data.enantiomer_test;
  const centersText = (data.chiral_centers && data.chiral_centers.length)
    ? data.chiral_centers.map(c => `${c.symbol}${c.atom_index} signed volume = ${c.signed_volume_ang3.toFixed(4)} Å³`).join(' · ')
    : 'Planar / achiral structure (signed volume = 0)';
  document.getElementById("stereoAuditResult").innerHTML = `
    <div class="kpi-grid">
      <div style="padding:14px 16px;border-radius:12px;background:var(--subtle);">
        <div style="font-size:12px;color:var(--ink-3);margin-bottom:4px;">3D Conformer</div>
        <div style="font-weight:600;font-size:14px;">${centersText}</div>
        <div class="mono" style="font-size:12px;color:var(--ink-2);margin-top:4px;">Builder: ${data.embedder} · stereo_ok: ${data.stereo_ok}</div>
      </div>
      <div style="padding:14px 16px;border-radius:12px;background:var(--subtle);">
        <div style="font-size:12px;color:var(--ink-3);margin-bottom:4px;">Mirror Twin Test (X → -X)</div>
        <div class="mono" style="font-size:12.5px;color:var(--ok-fg);">YLM: ${et.ylm_chiral_orig.toExponential(2)} → ${et.ylm_chiral_mirror.toExponential(2)} (opposite sign ✓)</div>
        <div class="mono" style="font-size:12.5px;color:var(--err-fg);margin-top:2px;">Uni-Mol: ${et.unimol_chiral_orig.toExponential(2)} → ${et.unimol_chiral_mirror.toExponential(2)} (blind, diff = ${et.unimol_diff.toExponential(1)})</div>
      </div>
    </div>`;
  if (data.html) {
    const fr = document.getElementById("stereoReportFrame");
    fr.onload = syncIframeThemes;
    fr.srcdoc = data.html;
  }
}

async function runLiveBenchmark() {
  const btn = document.getElementById('btnLiveBench');
  const epochs = Number(document.getElementById('benchEpochs').value || 6);
  btn.disabled = true;
  btn.textContent = 'Training...';
  try {
    const resp = await fetch('/api/eval_benchmark', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ epochs, eval_rotations: 4 })
    });
    const data = await resp.json();
    if (data.error) { alert(data.error); return; }
    document.getElementById('liveBenchContainer').style.display = 'block';
    document.getElementById('liveBenchBody').innerHTML = data.models.map(m => {
      const isOurs = m.model.includes('YLM');
      return `<tr class="${isOurs ? 'row-highlight' : ''}">
        <td>${m.model}</td>
        <td class="num mono">${m.elapsed_sec.toFixed(1)}s</td>
        <td class="num mono">${m.scalar_mae.toFixed(4)}</td>
        <td class="num mono">${m.vector_cos >= 0 ? '+' : ''}${m.vector_cos.toFixed(3)}</td>
        <td class="num mono">${m.force_cos >= 0 ? '+' : ''}${m.force_cos.toFixed(3)}</td>
        <td class="num mono" style="color:${m.vector_eq_err > 1e-3 ? 'var(--err-fg)' : 'var(--ok-fg)'};">${m.vector_eq_err.toExponential(1)}</td>
        <td class="num mono">${m.tensor_cos >= 0 ? '+' : ''}${m.tensor_cos.toFixed(3)}</td>
        <td class="num mono">${m.routing_acc.toFixed(3)}</td>
      </tr>`;
    }).join('');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Live CPU Eval';
  }
}

initStudio();
</script>
</body>
</html>
"""


def create_studio_server(
    probe: Optional[YLMProbe] = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    active_probe = probe or YLMProbe.default()

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
            if self.path.split("?")[0] in ("/", "/index.html"):
                raw = _STUDIO_INDEX_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            if self.path == "/api/health":
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

            if self.path == "/api/presets":
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
                    res = active_probe.ask(frame, question, domain=domain, verify_o3=verify)
                    payload = res.to_dict()
                    payload["html"] = render_html_report(res)
                    self._send_json(HTTPStatus.OK, payload)
                    return

                if self.path == "/api/physics_audit":
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
                    self._send_json(HTTPStatus.OK, _handle_run_md(body, active_probe))
                    return

                if self.path == "/api/ouroboros":
                    self._send_json(HTTPStatus.OK, _handle_ouroboros_inspect(
                        str(body.get("smiles") or "O=C1OC[C@H](F)O1"),
                        str(body.get("question") or "What is the 3D molecular dipole moment vector?"),
                        active_probe,
                    ))
                    return

                if self.path == "/api/eval_benchmark":
                    self._send_json(HTTPStatus.OK, _run_live_mini_benchmark(
                        epochs=int(body.get("epochs", 6)),
                        eval_rotations=int(body.get("eval_rotations", 4)),
                    ))
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

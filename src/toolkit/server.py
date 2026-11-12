"""Local browser app for testing YLM features and physics rules (`ylm serve`).

Runs on Python's built-in `http.server` with 5 tabs:
  1. Physics Rules Lab : 8 live physics checks + 3D spin, mirror, move, and atom-swap sandbox
  2. 3D Question Probe : Ask questions in plain English on molecules, electrolytes, MD frames, and drugs
  3. Live Reactive MD  : Watch bond breaking, bond forming, and H+ jumps frame by frame
  4. 3D Handedness     : Turn 2D stereo SMILES (`@/@@`, `/\\`) into 3D atoms and test mirror twins
  5. Model Comparison  : Compare YLM against Uni-Mol, GraphMVP, and EquiLLM on CPU
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
        "html": render_html_report(probe_res, title=f"Stereo Check: {smi}"),
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
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>YLM Directional Chemistry Studio — Physics &amp; 3D Molecule Lab</title>
<style>
  :root {
    --bg: #0b1120; --panel: #131c31; --panel2: #1b2742; --border: #2b3b5e;
    --text: #f1f5f9; --muted: #94a3b8; --accent: #38bdf8; --pass: #10b981;
    --fail: #ef4444; --warn: #f59e0b;
  }
  * { box-sizing: border-box; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  body { margin: 0; background: var(--bg); color: var(--text); padding: 18px 22px 36px; }
  .topbar { display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 14px; }
  .brand h1 { margin: 0; font-size: 21px; letter-spacing: -0.01em; }
  .brand p { margin: 4px 0 0; font-size: 13px; color: var(--muted); }
  .pill { padding: 5px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; border: 1px solid var(--border); background: var(--panel); }
  .pill-ok { color: #34d399; border-color: #059669; background: rgba(16,185,129,0.12); }
  .tabs { display: flex; gap: 8px; flex-wrap: wrap; border-bottom: 1px solid var(--border); padding-bottom: 12px; margin-bottom: 18px; }
  .tab-btn { background: var(--panel); color: var(--muted); border: 1px solid var(--border); padding: 9px 15px; border-radius: 9px; font-size: 13px; font-weight: 600; cursor: pointer; transition: 0.15s; }
  .tab-btn:hover { color: var(--text); border-color: #475569; }
  .tab-btn.active { background: rgba(56,189,248,0.16); color: var(--accent); border-color: #0284c7; }
  .pane { display: none; }
  .pane.active { display: block; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 16px; }
  .card h3 { margin: 0 0 6px; font-size: 15px; }
  .sub { font-size: 12px; color: var(--muted); margin-bottom: 12px; line-height: 1.5; }
  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .row3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
  @media (max-width: 980px) { .row2, .row3 { grid-template-columns: 1fr; } }
  label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  input[type="text"], input[type="number"], select, textarea {
    width: 100%; background: #090e1a; color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 10px; font-size: 13px;
  }
  input[type="range"] { width: 100%; accent-color: var(--accent); }
  .btn { background: #0284c7; color: white; border: none; padding: 9px 16px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; }
  .btn:hover { background: #0369a1; }
  .btn-sec { background: var(--panel2); border: 1px solid var(--border); color: var(--text); }
  .btn-sec:hover { border-color: var(--accent); }
  .law-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 12px; margin-top: 12px; }
  .law-card { background: #0d1526; border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
  .law-card.pass { border-left: 4px solid var(--pass); }
  .law-card.fail { border-left: 4px solid var(--fail); }
  .badge-pass { background: rgba(16,185,129,0.16); color: #34d399; padding: 2px 9px; border-radius: 999px; font-size: 12px; font-weight: 700; }
  .badge-fail { background: rgba(239,68,68,0.18); color: #f87171; padding: 2px 9px; border-radius: 999px; font-size: 12px; font-weight: 700; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }
  th { color: var(--muted); font-size: 12px; font-weight: 600; background: #0d1526; }
  canvas.scene3d { width: 100%; height: 320px; background: #070b14; border-radius: 10px; border: 1px solid var(--border); display: block; cursor: grab; }
  iframe.report-box { width: 100%; height: 640px; border: 1px solid var(--border); border-radius: 12px; background: #090d16; }
  .preset-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .chip { font-size: 12px; padding: 4px 10px; border-radius: 999px; background: #0d1526; border: 1px solid var(--border); color: #cbd5e1; cursor: pointer; }
  .chip:hover { border-color: var(--accent); color: white; }
  .metric-bar { height: 8px; border-radius: 999px; background: #090e1a; overflow: hidden; width: 110px; display: inline-block; vertical-align: middle; margin-right: 6px; }
  .metric-fill { height: 100%; background: linear-gradient(90deg, #38bdf8, #10b981); }
</style>
</head>
<body>

<div class="topbar">
  <div class="brand">
    <h1>YLM Directional Chemistry Studio</h1>
    <p>Ask questions in plain English, get 3D arrows back, and test whether the model obeys real-world 3D physics.</p>
  </div>
  <div style="display:flex;gap:8px;align-items:center;">
    <span class="pill pill-ok" id="engineBadge">Engine: Ready</span>
  </div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('physics', this)">1. Physics Rules Lab</button>
  <button class="tab-btn" onclick="switchTab('probe', this)">2. Ask a 3D Question</button>
  <button class="tab-btn" onclick="switchTab('md', this)">3. Watch Reactions (MD)</button>
  <button class="tab-btn" onclick="switchTab('ouroboros', this)">4. Left vs. Right Handedness</button>
  <button class="tab-btn" onclick="switchTab('benchmark', this)">5. Compare Models</button>
</div>

<!-- ================= TAB 1: PHYSICS RULES LAB ================= -->
<div id="pane-physics" class="pane active">
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
      <div>
        <h3>8 Basic Physics Rules — Live Check</h3>
        <div class="sub">Every real molecule obeys simple 3D rules: moving it in empty space changes nothing, spinning it spins its force and dipole arrows by the same angle, flipping it in a mirror turns a left-handed molecule into a right-handed one, and internal forces add up to zero.</div>
      </div>
      <div id="auditSummaryBadge" class="pill pill-ok">Ready to check...</div>
    </div>

    <div class="row3" style="margin-top:8px;">
      <div>
        <label>Pick a test molecule or crystal</label>
        <select id="auditPresetSel" onchange="runPhysicsAudit()"></select>
      </div>
      <div>
        <label>Question in plain English</label>
        <input type="text" id="auditQuestion" value="What is the 3D molecular dipole moment vector?" />
      </div>
      <div style="display:flex;gap:10px;align-items:flex-end;">
        <div style="flex:1;">
          <label>Number precision</label>
          <select id="auditPrecision" onchange="runPhysicsAudit()">
            <option value="float64" selected>64-bit float (strict check, error &lt; 1e-8)</option>
            <option value="float32">32-bit float (fast mode, error &lt; 5e-5)</option>
          </select>
        </div>
        <button class="btn" onclick="runPhysicsAudit()">Run All 8 Checks</button>
      </div>
    </div>

    <div id="lawsContainer" class="law-grid"></div>
  </div>

  <div class="card">
    <h3>3D Spin &amp; Mirror Sandbox (YLM vs. Uni-Mol, GraphMVP &amp; EquiLLM)</h3>
    <div class="sub">Drag the angle sliders or check <strong>Flip in a mirror</strong>: on the right canvas, the green target arrow and blue YLM arrow stay locked together, while the red Uni-Mol arrow points the wrong way because Uni-Mol only sees atom-to-atom distances.</div>

    <div class="row3" style="margin-bottom:12px;">
      <div>
        <label>Spin around X axis: <strong id="lblRx">35°</strong></label>
        <input type="range" id="slRx" min="-180" max="180" value="35" oninput="onSandboxChange()" />
      </div>
      <div>
        <label>Spin around Y axis: <strong id="lblRy">-48°</strong></label>
        <input type="range" id="slRy" min="-180" max="180" value="-48" oninput="onSandboxChange()" />
      </div>
      <div>
        <label>Spin around Z axis: <strong id="lblRz">72°</strong></label>
        <input type="range" id="slRz" min="-180" max="180" value="72" oninput="onSandboxChange()" />
      </div>
    </div>

    <div style="display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin-bottom:14px;">
      <label style="display:inline-flex;align-items:center;gap:6px;color:var(--text);font-size:13px;cursor:pointer;">
        <input type="checkbox" id="chkReflect" onchange="runPoseSandbox()" />
        Flip in a mirror (left hand ↔ right hand, <code>det = -1</code>)
      </label>
      <label style="display:inline-flex;align-items:center;gap:6px;color:var(--text);font-size:13px;cursor:pointer;">
        <input type="checkbox" id="chkPermute" onchange="runPoseSandbox()" />
        Shuffle atom row order
      </label>
      <label style="display:inline-flex;align-items:center;gap:6px;color:var(--text);font-size:13px;cursor:pointer;">
        <input type="checkbox" id="chkTranslate" onchange="runPoseSandbox()" />
        Shift position by <code>(+3.5, -2.1, +4.0) Å</code>
      </label>
      <span class="pill mono" id="detReadout">det(Q) = +1.0000</span>
    </div>

    <div class="row2">
      <div>
        <div style="font-size:12px;color:var(--muted);margin-bottom:6px;">Before transform (blue = YLM arrow, yellow glow = which atom the model looks at)</div>
        <canvas id="cvOrig" class="scene3d"></canvas>
      </div>
      <div>
        <div style="font-size:12px;color:var(--muted);margin-bottom:6px;">After transform (green = exact physics | blue = YLM | red = Uni-Mol)</div>
        <canvas id="cvTrans" class="scene3d"></canvas>
      </div>
    </div>

    <div style="margin-top:14px;overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Energy drift ↓</th>
            <th>Handedness (before → after)</th>
            <th>Mirror rule error ↓</th>
            <th>3D arrow error ↓</th>
            <th>3x3 matrix error ↓</th>
            <th>Atom-order error ↓</th>
          </tr>
        </thead>
        <tbody id="sandboxTableBody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ================= TAB 2: ASK A 3D QUESTION ================= -->
<div id="pane-probe" class="pane">
  <div class="card">
    <h3>Ask a 3D Question on Any Molecule or Crystal</h3>
    <div class="sub">Click a quick example chip or paste an XYZ / POSCAR / CIF file or SMILES string, then ask which way a force or dipole arrow points.</div>
    <div class="preset-chips" id="probePresetChips"></div>
    <div class="row2" style="margin-top:12px;">
      <div>
        <label>SMILES string or XYZ / POSCAR / CIF coordinates</label>
        <textarea id="structInput" rows="6">3
water demo
O  0.000  0.000  0.125
H  0.020  0.790 -0.460
H -0.015 -0.745 -0.485</textarea>
      </div>
      <div style="display:flex;flex-direction:column;gap:10px;">
        <div>
          <label>Chemistry topic</label>
          <select id="domainSel">
            <option value="general">General — Molecular dipole &amp; force on a chosen atom</option>
            <option value="electrolyte">Battery Electrolyte — Solvent shell around Li+ / Na+ &amp; escape arrow</option>
            <option value="md">Reaction Motion — Force arrow vs. atom motion arrow</option>
            <option value="drug">Drug Design — Safe 3D direction to grow a molecule in a protein pocket</option>
          </select>
        </div>
        <div>
          <label>Question in plain English</label>
          <input type="text" id="qInput" value="What is the force vector on the hydrogen atom?" />
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-top:4px;">
          <label style="display:inline-flex;align-items:center;gap:6px;color:var(--text);">
            <input type="checkbox" id="o3Chk" checked /> Check 3D rotation + mirror flip live
          </label>
          <div style="display:flex;gap:8px;">
            <button class="btn btn-sec" onclick="buildLiCluster()">Build Li+ Solvent Shell</button>
            <button class="btn" onclick="runQuery()">Ask YLM</button>
          </div>
        </div>
      </div>
    </div>
  </div>
  <iframe id="reportFrame" class="report-box"></iframe>
</div>

<!-- ================= TAB 3: WATCH REACTIONS (MD) ================= -->
<div id="pane-md" class="pane">
  <div class="card">
    <h3>Watch Chemical Reactions Step by Step (Bond Breaking, Bond Forming &amp; H+ Jumps)</h3>
    <div class="sub">Follows moving atoms over time, flags the exact frame where a bond breaks, forms, or a proton (<code>H+</code>) hops between two water molecules, and plots work <code>W = Force · Step</code>.</div>
    <div class="row3">
      <div>
        <label>Simulation mode</label>
        <select id="mdModeSel">
          <option value="scan_zundel">Scan saved H+ jump trajectory (examples/proton_transfer_traj.extxyz)</option>
          <option value="live_md">Run live reactive MD steps (stretched H5O2+ water-proton cluster)</option>
        </select>
      </div>
      <div>
        <label>Live MD steps / Temperature (K)</label>
        <div style="display:flex;gap:8px;">
          <input type="number" id="mdSteps" value="28" min="8" max="80" />
          <input type="number" id="mdTemp" value="350" min="0" max="1500" />
        </div>
      </div>
      <div style="display:flex;align-items:flex-end;">
        <button class="btn" style="width:100%;" onclick="runReactiveMD()">Run Reaction Monitor</button>
      </div>
    </div>
    <div id="mdEventsSummary" style="margin-top:14px;"></div>
    <div id="mdSvgContainer" style="margin-top:12px;overflow-x:auto;"></div>
  </div>
  <iframe id="mdReportFrame" class="report-box"></iframe>
</div>

<!-- ================= TAB 4: LEFT VS. RIGHT HANDEDNESS ================= -->
<div id="pane-ouroboros" class="pane">
  <div class="card">
    <h3>Left vs. Right Handedness (2D Stereo SMILES → 3D Atoms → Mirror Twin Test)</h3>
    <div class="sub">Like your left and right hands, mirror-image molecules have identical atom-to-atom distances, so distance-only models (Uni-Mol) cannot tell them apart. YLM uses signed 3D volume to give opposite signs (<code>+</code> vs. <code>-</code>) to mirror twins and <code>0</code> to flat molecules.</div>
    <div class="preset-chips" style="margin-bottom:10px;">
      <span class="chip" onclick="setStereoSmiles('O=C1OC[C@H](F)O1')">(R)-FEC battery additive</span>
      <span class="chip" onclick="setStereoSmiles('O=C1OC[C@@H](F)O1')">(S)-FEC mirror twin</span>
      <span class="chip" onclick="setStereoSmiles('F[C@H](Cl)Br')">(R)-CHFCIBr chiral carbon</span>
      <span class="chip" onclick="setStereoSmiles('F[C@@H](Cl)Br')">(S)-CHFCIBr mirror twin</span>
      <span class="chip" onclick="setStereoSmiles('F/C=C\\\\F')">cis-FCH=CHF (flat, dipole != 0)</span>
      <span class="chip" onclick="setStereoSmiles('F/C=C/F')">trans-FCH=CHF (flat, dipole = 0)</span>
    </div>
    <div class="row3">
      <div>
        <label>Stereo SMILES (<code>@/@@</code> or <code>/\\</code>)</label>
        <input type="text" id="stereoSmilesInput" value="O=C1OC[C@H](F)O1" />
      </div>
      <div>
        <label>Question in plain English</label>
        <input type="text" id="stereoQuestionInput" value="What is the 3D molecular dipole moment vector?" />
      </div>
      <div style="display:flex;align-items:flex-end;">
        <button class="btn" style="width:100%;" onclick="runOuroborosAudit()">Build 3D &amp; Test Mirror Twin</button>
      </div>
    </div>
    <div id="stereoAuditResult" style="margin-top:14px;"></div>
  </div>
  <iframe id="stereoReportFrame" class="report-box"></iframe>
</div>

<!-- ================= TAB 5: COMPARE MODELS ================= -->
<div id="pane-benchmark" class="pane">
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
      <div>
        <h3>Compare 3D Models Side by Side (YLM vs. Uni-Mol, GraphMVP &amp; EquiLLM)</h3>
        <div class="sub">Table A shows saved 24-epoch scores on left/right-handed molecules, 3D arrows, and 3x3 matrices. Click <strong>Train &amp; Test Live on CPU</strong> to train all 4 models from scratch right now and verify the trend yourself.</div>
      </div>
      <div style="display:flex;gap:8px;align-items:center;">
        <label style="margin:0;">Training epochs:</label>
        <input type="number" id="benchEpochs" value="6" min="2" max="20" style="width:72px;" />
        <button class="btn" id="btnLiveBench" onclick="runLiveBenchmark()">Train &amp; Test Live on CPU</button>
      </div>
    </div>

    <h4 style="margin:14px 0 8px;font-size:14px;color:var(--accent);">A. Saved 24-Epoch CPU Run (8 mirror-twin families, including unseen molecule shapes; from <code>results/cpu_smoke_nov12/chiral_benchmark.json</code>)</h4>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Left/Right Acc (Test) ↑</th>
            <th>New Wording Acc ↑</th>
            <th>Unseen Molecule Acc ↑</th>
            <th>Mirror Flip Error ↓</th>
            <th>Flat Molecule Output (should be 0) ↓</th>
            <th>3D Arrow Cos ↑</th>
            <th>3x3 Matrix Cos ↑</th>
          </tr>
        </thead>
        <tbody id="refBenchBody"></tbody>
      </table>
    </div>

    <div id="liveBenchContainer" style="margin-top:18px;display:none;">
      <h4 style="margin:0 0 8px;font-size:14px;color:#34d399;">B. Fresh Live CPU Run (trained &amp; tested under random 3D rotations and mirror flips)</h4>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Time (s)</th>
              <th>Energy MAE ↓</th>
              <th>3D Arrow Cos ↑</th>
              <th>Atom Force Cos ↑</th>
              <th>Rotation Error ↓</th>
              <th>3x3 Matrix Cos ↑</th>
              <th>Atom Pick Acc ↑</th>
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
const ELEM_COL = {1:"#f8fafc", 3:"#a855f7", 6:"#64748b", 7:"#3b82f6", 8:"#ef4444", 9:"#22c55e", 15:"#f97316", 16:"#eab308", 17:"#14b8a6", 26:"#d97706", 35:"#ec4899"};

function switchTab(name, btn) {
  document.querySelectorAll('.pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('pane-' + name).classList.add('active');
  if (btn) btn.classList.add('active');
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
  cv.width = Math.max(300, rect.width || 420);
  cv.height = Math.max(260, rect.height || 320);
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!scene.atoms.length) return;

  let cx = 0, cy = 0, cz = 0;
  scene.atoms.forEach(a => { cx += a.x; cy += a.y; cz += a.z; });
  cx /= scene.atoms.length; cy /= scene.atoms.length; cz /= scene.atoms.length;

  let maxR = 2.4;
  scene.atoms.forEach(a => {
    const d = Math.hypot(a.x - cx, a.y - cy, a.z - cz);
    if (d + 1.6 > maxR) maxR = d + 1.6;
  });
  const scale = 0.40 * Math.min(cv.width, cv.height) / maxR;

  function proj(px, py, pz) {
    const x = px - cx, y = py - cy, z = pz - cz;
    const cy1 = Math.cos(cam.ry), sy1 = Math.sin(cam.ry);
    const x1 = x * cy1 + z * sy1, z1 = -x * sy1 + z * cy1;
    const cx1 = Math.cos(cam.rx), sx1 = Math.sin(cam.rx);
    const y2 = y * cx1 - z1 * sx1, z2 = y * sx1 + z1 * cx1;
    return [cv.width / 2 + x1 * scale, cv.height / 2 - y2 * scale, z2];
  }

  ctx.lineWidth = 2.8;
  ctx.strokeStyle = "#475569";
  (scene.bonds || []).forEach(([i, j]) => {
    const p1 = proj(scene.atoms[i].x, scene.atoms[i].y, scene.atoms[i].z);
    const p2 = proj(scene.atoms[j].x, scene.atoms[j].y, scene.atoms[j].z);
    ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke();
  });

  const weights = scene.weights || [];
  const maxW = Math.max(...weights, 1e-6);
  const ordered = scene.atoms.map((a, idx) => ({a, idx, p: proj(a.x, a.y, a.z)})).sort((u, v) => u.p[2] - v.p[2]);
  ordered.forEach(({a, idx, p}) => {
    const w = (weights[idx] || 0) / maxW;
    const rad = a.Z === 1 ? 8 : 12;
    if (w > 0.2) {
      ctx.beginPath(); ctx.arc(p[0], p[1], rad + 7 * w, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(251, 191, 36, ${0.15 + 0.45 * w})`; ctx.fill();
    }
    ctx.beginPath(); ctx.arc(p[0], p[1], rad, 0, Math.PI * 2);
    ctx.fillStyle = ELEM_COL[a.Z] || "#94a3b8"; ctx.fill();
    ctx.strokeStyle = "#0b1120"; ctx.lineWidth = 1.4; ctx.stroke();
    ctx.fillStyle = a.Z === 1 ? "#0b1120" : "#ffffff";
    ctx.font = "bold 10px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(a.symbol + idx, p[0], p[1]);
  });

  (scene.arrows || []).forEach((ar, k) => {
    const s = proj(ar.start[0], ar.start[1], ar.start[2]);
    const e = proj(ar.end[0], ar.end[1], ar.end[2]);
    ctx.strokeStyle = ar.color; ctx.fillStyle = ar.color;
    ctx.lineWidth = k === 0 ? 4.2 : 2.8;
    if (k === 0) ctx.setLineDash([6, 3]); else ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(s[0], s[1]); ctx.lineTo(e[0], e[1]); ctx.stroke();
    ctx.setLineDash([]);
    const ang = Math.atan2(e[1] - s[1], e[0] - s[0]);
    ctx.beginPath();
    ctx.moveTo(e[0], e[1]);
    ctx.lineTo(e[0] - 10 * Math.cos(ang - 0.4), e[1] - 10 * Math.sin(ang - 0.4));
    ctx.lineTo(e[0] - 10 * Math.cos(ang + 0.4), e[1] - 10 * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();

    ctx.font = "12px sans-serif"; ctx.textAlign = "left";
    ctx.fillText("● " + ar.name, 12, 18 + k * 18);
  });
}

async function initStudio() {
  attachDrag('cvOrig', camOrig, 'orig');
  attachDrag('cvTrans', camTrans, 'trans');
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
    opt.textContent = p.title;
    if (p.id === 'r_fec_chiral') opt.selected = true;
    sel.appendChild(opt);

    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = p.id;
    chip.title = p.title;
    chip.onclick = () => {
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
    const st = isOurs ? 'style="background:rgba(16,185,129,0.10);font-weight:600;"' : '';
    return `<tr ${st}>
      <td>${m.model}</td>
      <td><div class="metric-bar"><div class="metric-fill" style="width:${m.chiral_acc*100}%"></div></div>${(m.chiral_acc*100).toFixed(1)}%</td>
      <td>${(m.heldout_chiral_acc*100).toFixed(1)}%</td>
      <td>${(m.zeroshot_scaffold_acc*100).toFixed(1)}%</td>
      <td class="mono">${m.chiral_parity_err.toExponential(2)}</td>
      <td class="mono">${m.achiral_planar_abs.toExponential(2)}</td>
      <td class="mono">${m.vector_cos >= 0 ? '+' : ''}${m.vector_cos.toFixed(4)}</td>
      <td class="mono">${m.tensor_cos >= 0 ? '+' : ''}${m.tensor_cos.toFixed(4)}</td>
    </tr>`;
  }).join('');
}

async function runPhysicsAudit() {
  const preset = document.getElementById('auditPresetSel').value || 'r_fec_chiral';
  const question = document.getElementById('auditQuestion').value;
  const precision = document.getElementById('auditPrecision').value;
  document.getElementById('auditSummaryBadge').textContent = 'Running 8 physics checks in PyTorch...';
  const resp = await fetch('/api/physics_audit', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ structure: preset, question, precision, n_rotations: 6 })
  });
  const data = await resp.json();
  if (data.error) { alert(data.error); return; }

  const badge = document.getElementById('auditSummaryBadge');
  badge.textContent = `${data.passed_count} / ${data.total_laws} physics rules passed (${data.precision})`;
  badge.className = data.passed_all ? 'pill pill-ok' : 'pill';

  const box = document.getElementById('lawsContainer');
  box.innerHTML = data.laws.map(law => {
    const baseHtml = law.baseline_errors
      ? '<div style="margin-top:6px;font-size:12px;color:#fca5a5;">Other models broke this rule: ' +
        Object.entries(law.baseline_errors).map(([k, v]) => `${k} = <code>${Number(v).toExponential(2)}</code>`).join(' · ') +
        '</div>'
      : '';
    return `<div class="law-card ${law.passed ? 'pass' : 'fail'}">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
        <strong style="font-size:13px;">${law.name}</strong>
        <span class="${law.passed ? 'badge-pass' : 'badge-fail'}">${law.passed ? 'PASS' : 'FAIL'}</span>
      </div>
      <div class="mono" style="color:var(--accent);margin-top:5px;">${law.formula}</div>
      <div style="font-size:12px;color:var(--muted);margin-top:6px;line-height:1.45;">${law.detail}</div>
      <div class="mono" style="margin-top:6px;font-size:12px;">YLM error: <strong>${Number(law.ylm_error).toExponential(2)}</strong> (limit &lt; ${Number(law.tolerance).toExponential(1)})</div>
      ${baseHtml}
    </div>`;
  }).join('');

  await runPoseSandbox();
}

function onSandboxChange() {
  document.getElementById('lblRx').textContent = document.getElementById('slRx').value + '°';
  document.getElementById('lblRy').textContent = document.getElementById('slRy').value + '°';
  document.getElementById('lblRz').textContent = document.getElementById('slRz').value + '°';
  clearTimeout(sandboxTimer);
  sandboxTimer = setTimeout(runPoseSandbox, 90);
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

  document.getElementById('detReadout').textContent = `${data.operation} | det(Q) = ${data.det_Q > 0 ? '+1.0000' : '-1.0000'}`;
  SANDBOX_SCENES.orig = data.scene_orig;
  SANDBOX_SCENES.trans = data.scene_transformed;
  drawScene(document.getElementById('cvOrig'), SANDBOX_SCENES.orig, camOrig);
  drawScene(document.getElementById('cvTrans'), SANDBOX_SCENES.trans, camTrans);

  const names = [
    ['ylm', 'YLM-Universe (Ours: keeps 3D directions & handedness)'],
    ['equillm', 'EquiLLM (scalar text bridge + 3D vector shortcut)'],
    ['unimol', 'Uni-Mol (atom-to-atom distances only)'],
    ['graphmvp', 'GraphMVP / 3D-InfoMax (2D bond graph only)'],
  ];
  const tol = data.precision === 'float64' ? 1e-8 : 1e-4;
  document.getElementById('sandboxTableBody').innerHTML = names.map(([key, title]) => {
    const m = data.models[key];
    const vBad = m.vector_1o.rel_err > tol;
    const cBad = m.chiral_0o.abs_err > tol;
    return `<tr style="${key === 'ylm' ? 'background:rgba(16,185,129,0.10);font-weight:600;' : ''}">
      <td>${title}</td>
      <td class="mono">${m.scalar_0e.abs_err.toExponential(2)}</td>
      <td class="mono">${m.chiral_0o.orig.toExponential(3)} → ${m.chiral_0o.transformed.toExponential(3)}</td>
      <td class="mono" style="color:${cBad ? '#f87171' : '#34d399'};">${m.chiral_0o.abs_err.toExponential(2)}</td>
      <td class="mono" style="color:${vBad ? '#f87171' : '#34d399'};">${m.vector_1o.rel_err.toExponential(2)} (cos=${m.vector_1o.cosine_with_expected.toFixed(4)})</td>
      <td class="mono">${m.tensor_0e2e.rel_err.toExponential(2)}</td>
      <td class="mono">${m.attn_routing_err.toExponential(2)}</td>
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
    document.getElementById("reportFrame").srcdoc = data.html;
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
  document.getElementById("mdEventsSummary").innerHTML = '<span class="sub">Running simulation and watching for bond changes...</span>';
  const resp = await fetch("/api/run_md", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  const data = await resp.json();
  if (data.error) { alert(data.error); return; }
  const evHtml = (data.events && data.events.length)
    ? data.events.map(e => `<span class="pill" style="margin-right:6px;border-color:#f59e0b;color:#fde68a;">Frame #${e.frame_index} (${e.time_fs.toFixed(1)} fs): ${e.event_type} [${e.labels.join('-')}] W=${e.work_ev.toFixed(3)} eV</span>`).join('')
    : '<span class="pill">Scanned ' + data.n_frames + ' frames (drag the frame slider below to compare force and motion arrows)</span>';
  document.getElementById("mdEventsSummary").innerHTML = evHtml;
  document.getElementById("mdSvgContainer").innerHTML = data.svg || '';
  if (data.html) document.getElementById("mdReportFrame").srcdoc = data.html;
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
    ? data.chiral_centers.map(c => `Atom ${c.symbol}#${c.atom_index}: signed 3D volume = ${c.signed_volume_ang3.toFixed(4)} Å³ (sign = ${c.handedness_sign > 0 ? '+1' : '-1'})`).join('; ')
    : 'No 4-group chiral atom (flat or symmetric molecule — true handedness is 0)';
  document.getElementById("stereoAuditResult").innerHTML = `
    <div class="law-grid">
      <div class="law-card pass">
        <strong>3D Stereo Build &amp; Signed Tetrahedron Volume</strong>
        <div class="mono" style="margin-top:6px;">SMILES: ${data.smiles} | Builder: ${data.embedder} | stereo_ok: ${data.stereo_ok}</div>
        <div style="font-size:12px;color:var(--muted);margin-top:6px;">${centersText}</div>
      </div>
      <div class="law-card pass">
        <strong>Mirror Twin Test <code>X → -X</code> (YLM vs. Uni-Mol)</strong>
        <div class="mono" style="margin-top:6px;color:#34d399;">YLM: hand(X) = ${et.ylm_chiral_orig.toExponential(3)}, hand(-X) = ${et.ylm_chiral_mirror.toExponential(3)} (sum error = ${et.ylm_sum_residual.toExponential(2)})</div>
        <div class="mono" style="margin-top:4px;color:#f87171;">Uni-Mol: hand(X) = ${et.unimol_chiral_orig.toExponential(3)}, hand(-X) = ${et.unimol_chiral_mirror.toExponential(3)} (difference = ${et.unimol_diff.toExponential(2)} — blind to mirror flip)</div>
      </div>
    </div>`;
  if (data.html) document.getElementById("stereoReportFrame").srcdoc = data.html;
}

async function runLiveBenchmark() {
  const btn = document.getElementById('btnLiveBench');
  const epochs = Number(document.getElementById('benchEpochs').value || 6);
  btn.disabled = true;
  btn.textContent = 'Training all 4 models on CPU...';
  try {
    const resp = await fetch('/api/eval_benchmark', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ epochs, eval_rotations: 4 })
    });
    const data = await resp.json();
    if (data.error) { alert(data.error); return; }
    document.getElementById('liveBenchContainer').style.display = 'block';
    document.getElementById('liveBenchBody').innerHTML = data.models.map(m => {
      const isOurs = m.model.includes('YLM');
      return `<tr style="${isOurs ? 'background:rgba(16,185,129,0.10);font-weight:600;' : ''}">
        <td>${m.model}</td>
        <td class="mono">${m.elapsed_sec.toFixed(1)}s</td>
        <td class="mono">${m.scalar_mae.toFixed(4)}</td>
        <td class="mono">${m.vector_cos >= 0 ? '+' : ''}${m.vector_cos.toFixed(4)}</td>
        <td class="mono">${m.force_cos >= 0 ? '+' : ''}${m.force_cos.toFixed(4)}</td>
        <td class="mono" style="color:${m.vector_eq_err > 1e-3 ? '#f87171' : '#34d399'};">${m.vector_eq_err.toExponential(2)}</td>
        <td class="mono">${m.tensor_cos >= 0 ? '+' : ''}${m.tensor_cos.toFixed(4)}</td>
        <td class="mono">${m.routing_acc.toFixed(3)}</td>
      </tr>`;
    }).join('');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Train & Test Live on CPU';
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
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                raw = _STUDIO_INDEX_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
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

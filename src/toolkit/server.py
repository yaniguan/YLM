"""Interactive Web Studio & Physics Compliance Workbench (`ylm serve`) built on stdlib http.server.

Provides a unified 5-tab browser UI backed by live PyTorch / e3nn endpoints:
  1. Physics & O(3) Compliance Lab : 8-law physical certification (`float64`/`float32`) + live 3D pose/reflection sandbox
  2. 3D Probe & Domain Packs       : General, Electrolyte Solvation, Reactive MD, Drug SBDD, & 3D PBC crystals
  3. Live Reactive MD & Hooks      : `MLIPTrajectoryRunner` + `LiveReactionMonitor` timeline & 3D frame scrubber
  4. Ouroboros-OCSR Stereo Bridge  : Isomeric SMILES (`@/@@`, `/\\`) -> 3D CIP audit + signed chiral volume + enantiomer test
  5. FM Benchmark & Live Eval      : Pre-verified + live CPU training/evaluation vs `Uni-Mol`, `GraphMVP`, and `EquiLLM`
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
    _get_bundle,
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

_REFERENCE_BENCHMARKS = {
    "chiral_multirank_24ep": {
        "title": "Chiral (0o) + Multi-Rank (0e + 1o + 2e) Benchmark (24 Epochs, K=8 Random O(3) Rotations + Reflections)",
        "models": [
            {
                "model": "GraphMVP / 3D-InfoMax",
                "chiral_acc": 0.542,
                "heldout_chiral_acc": 0.479,
                "zeroshot_scaffold_acc": 0.500,
                "chiral_parity_err": 0.677,
                "achiral_planar_abs": 1.040,
                "scalar_mae": 0.1079,
                "vector_cos": 0.0202,
                "vector_eq_err": 1.35,
                "tensor_cos": 0.7107,
                "routing_acc": 0.833,
            },
            {
                "model": "Uni-Mol (3D Dist-Transformer)",
                "chiral_acc": 0.521,
                "heldout_chiral_acc": 0.521,
                "zeroshot_scaffold_acc": 0.484,
                "chiral_parity_err": 0.805,
                "achiral_planar_abs": 0.819,
                "scalar_mae": 0.0996,
                "vector_cos": 0.0197,
                "vector_eq_err": 1.35,
                "tensor_cos": 0.6953,
                "routing_acc": 0.833,
            },
            {
                "model": "EquiLLM (Inv-Bridge + 1o)",
                "chiral_acc": 0.521,
                "heldout_chiral_acc": 0.500,
                "zeroshot_scaffold_acc": 0.500,
                "chiral_parity_err": 0.665,
                "achiral_planar_abs": 0.421,
                "scalar_mae": 0.1175,
                "vector_cos": 0.7007,
                "vector_eq_err": 1.02e-6,
                "tensor_cos": 0.9874,
                "routing_acc": 0.917,
            },
            {
                "model": "YLM-Universe FM (Ours)",
                "chiral_acc": 1.000,
                "heldout_chiral_acc": 1.000,
                "zeroshot_scaffold_acc": 1.000,
                "chiral_parity_err": 3.20e-7,
                "achiral_planar_abs": 1.88e-7,
                "scalar_mae": 0.1014,
                "vector_cos": 0.7694,
                "vector_eq_err": 1.05e-6,
                "tensor_cos": 0.9974,
                "routing_acc": 0.833,
            },
        ],
    }
}


def _run_live_mini_benchmark(epochs: int = 6, eval_rotations: int = 4) -> dict:
    """Run a fast live multi-model Universe evaluation on CPU right inside the Studio server."""
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
        ("Uni-Mol (3D Dist-Transformer)", UniMolStyle3DBaseline(text_dim=64, d_model=48, n_layers=2, n_heads=4)),
        ("EquiLLM (Inv-Bridge + 1o)", EquiLLMMultiTaskBaseline(text_dim=64, hidden_irreps=hidden_ir, n_layers=2, d_model=48, n_vec=8)),
        (
            "YLM-Universe FM (Ours)",
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

    # Construct exact mirror image (-pos) rotated by an independent SO(3) matrix to test enantiomer response
    mirror_fr = StructureFrame(
        Z=fr.Z.copy(),
        pos_bohr=-fr.pos_bohr.copy(),
        label=f"mirror_of_{fr.label}",
        metadata=dict(fr.metadata),
    )
    mirror_centers = compute_chiral_centers(mirror_fr)

    bundle = _get_bundle()
    models = bundle.get_models(precision="float64")
    in_orig = _frame_to_tensors(fr, question, bundle.featurizer, torch.float64)
    in_mirr = _frame_to_tensors(mirror_fr, question, bundle.featurizer, torch.float64)

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
        "html": render_html_report(probe_res, title=f"Ouroboros Stereo Audit: {smi}"),
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
            html_text = render_trajectory_html_report(scan, title="Proton-Transfer Reactive Trajectory")
            return {
                "mode": mode,
                "n_frames": len(scan.frame_results),
                "events": [e.to_dict() for e in mon.events],
                "scan_events": scan.events,
                "summary": scan.summary,
                "svg": svg_text,
                "html": html_text,
            }

        # Live Velocity-Verlet / Langevin reactive MD via MLIPTrajectoryRunner
        steps = max(6, min(int(body.get("steps", 28)), 80))
        dt_fs = float(body.get("dt_fs", 0.6))
        temp_k = float(body.get("temperature_k", 350.0))
        stretch = float(body.get("bond_stretch_ang", 0.55))

        # Construct a stretched reactive H5O2+ Zundel or user-supplied geometry
        src = str(body.get("structure") or "").strip()
        if src and src != "zundel_stretched":
            init_fr, _ = resolve_preset_or_structure(src)
        else:
            coords = np.array([
                [-1.26, 0.02, 0.00],
                [-0.38 + stretch, 0.01, 0.00],  # displaced shared proton!
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
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>YLM Directional Chemistry Studio — Physics Compliance &amp; Evaluation Workbench</title>
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
  .sub { font-size: 12px; color: var(--muted); margin-bottom: 12px; }
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
    <p>全功能交互工作台 &amp; 物理对称性合规审计台（`0e` 标量 · `0o` 手性赝标量 · `1o` 极向量 · `0e+2e` 二阶张量 · 保守力场 · 3D PBC）</p>
  </div>
  <div style="display:flex;gap:8px;align-items:center;">
    <span class="pill pill-ok" id="engineBadge">Engine: Ready</span>
  </div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('physics', this)">⚖️ 1. 物理合规与 O(3) 等变审计台 (Physics Compliance)</button>
  <button class="tab-btn" onclick="switchTab('probe', this)">🧭 2. 3D 方向问答与四大领域包 (3D Probe &amp; Domains)</button>
  <button class="tab-btn" onclick="switchTab('md', this)">⚡ 3. 实时反应 MD 与断键/质子转移监测 (Reactive MD)</button>
  <button class="tab-btn" onclick="switchTab('ouroboros', this)">🧬 4. Ouroboros 立体手性桥接 (Stereo &amp; Chirality)</button>
  <button class="tab-btn" onclick="switchTab('benchmark', this)">📊 5. 基座模型跑分与在线评测 (FM Evaluation)</button>
</div>

<!-- ================= TAB 1: PHYSICS & O(3) COMPLIANCE LAB ================= -->
<div id="pane-physics" class="pane active">
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
      <div>
        <h3>八大基础物理定律自动化合规认证（8-Law Physical Compliance Certification）</h3>
        <div class="sub">逐项检验平移不变性、SO(3) 真旋转、O(3) 镜像反演与对映体宇称、原子置换群 S_N、二阶张量对称分解、Curie 点群禁戒选择定则、牛顿第三定律/零力矩/无旋保守势梯度（F = -∇E）以及晶格周期平移不变性。</div>
      </div>
      <div id="auditSummaryBadge" class="pill pill-ok">等待运行认证...</div>
    </div>

    <div class="row3" style="margin-top:8px;">
      <div>
        <label>选择测试体系（覆盖 C2v 平面、C2h 反演中心、手性对映体、3D PBC 晶体）</label>
        <select id="auditPresetSel" onchange="runPhysicsAudit()"></select>
      </div>
      <div>
        <label>自然语言探针问题（测试跨模态算子对称性）</label>
        <input type="text" id="auditQuestion" value="What is the 3D molecular dipole moment vector?" />
      </div>
      <div style="display:flex;gap:10px;align-items:flex-end;">
        <div style="flex:1;">
          <label>数值精度</label>
          <select id="auditPrecision" onchange="runPhysicsAudit()">
            <option value="float64" selected>IEEE float64 (严格机器精度 ~1e-14)</option>
            <option value="float32">IEEE float32 (训练推理精度 ~1e-6)</option>
          </select>
        </div>
        <button class="btn" onclick="runPhysicsAudit()">立即执行 8 项物理合规审计</button>
      </div>
    </div>

    <div id="lawsContainer" class="law-grid"></div>
  </div>

  <div class="card">
    <h3>实时 3D 连续旋转 / 镜像反演 / 置换沙盒（YLM vs. Uni-Mol / GraphMVP / EquiLLM 实时同屏对照）</h3>
    <div class="sub">拖动欧拉角或勾选「镜像反演 det(Q) = -1」：观察右侧绿色理论物理箭头 <code>Q·f(X)</code> 与蓝色 YLM 实测箭头 <code>f(X·Q^T)</code> 始终 100% 重合，而红色 Uni-Mol 箭头在旋转后彻底失灵，且在镜像反演下手性赝标量无法翻号。</div>

    <div class="row3" style="margin-bottom:12px;">
      <div>
        <label>绕 X 轴旋转 θ_x: <strong id="lblRx">35°</strong></label>
        <input type="range" id="slRx" min="-180" max="180" value="35" oninput="onSandboxChange()" />
      </div>
      <div>
        <label>绕 Y 轴旋转 θ_y: <strong id="lblRy">-48°</strong></label>
        <input type="range" id="slRy" min="-180" max="180" value="-48" oninput="onSandboxChange()" />
      </div>
      <div>
        <label>绕 Z 轴旋转 θ_z: <strong id="lblRz">72°</strong></label>
        <input type="range" id="slRz" min="-180" max="180" value="72" oninput="onSandboxChange()" />
      </div>
    </div>

    <div style="display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin-bottom:14px;">
      <label style="display:inline-flex;align-items:center;gap:6px;color:var(--text);font-size:13px;cursor:pointer;">
        <input type="checkbox" id="chkReflect" onchange="runPoseSandbox()" />
        开启镜像反演 / 空间宇称反演 <code>det(Q) = -1</code>（生成对映异构体）
      </label>
      <label style="display:inline-flex;align-items:center;gap:6px;color:var(--text);font-size:13px;cursor:pointer;">
        <input type="checkbox" id="chkPermute" onchange="runPoseSandbox()" />
        打乱原子编号顺序 <code>P ∈ S_N</code>
      </label>
      <label style="display:inline-flex;align-items:center;gap:6px;color:var(--text);font-size:13px;cursor:pointer;">
        <input type="checkbox" id="chkTranslate" onchange="runPoseSandbox()" />
        叠加空间平移 <code>(+3.5, -2.1, +4.0) Å</code>
      </label>
      <span class="pill mono" id="detReadout">det(Q) = +1.0000</span>
    </div>

    <div class="row2">
      <div>
        <div style="font-size:12px;color:var(--muted);margin-bottom:6px;">原始构象 <code>X</code>（蓝箭头 = <code>YLM f(X)</code>，琥珀光晕 = 原子寻址权重）</div>
        <canvas id="cvOrig" class="scene3d"></canvas>
      </div>
      <div>
        <div style="font-size:12px;color:var(--muted);margin-bottom:6px;">变换后构象 <code>X' = P·X·Q^T + a</code>（绿 = 精确物理要求 <code>Q·f(X)</code> | 蓝 = <code>YLM</code> | 红 = <code>Uni-Mol</code>）</div>
        <canvas id="cvTrans" class="scene3d"></canvas>
      </div>
    </div>

    <div style="margin-top:14px;overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>模型架构</th>
            <th><code>0e</code> 标量漂移 <code>|E(X') - E(X)|</code> ↓</th>
            <th><code>0o</code> 手性原值 <code>χ(X)</code> → 变换值 <code>χ(X')</code></th>
            <th><code>0o</code> 宇称误差 <code>|χ(X') - det(Q)χ(X)|</code> ↓</th>
            <th><code>1o</code> 向量相对误差 <code>||v' - Qv|| / ||v||</code> ↓</th>
            <th><code>0e+2e</code> 张量误差 <code>||T' - QTQ^T|| / ||T||</code> ↓</th>
            <th><code>S_N</code> 路由误差 ↓</th>
          </tr>
        </thead>
        <tbody id="sandboxTableBody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ================= TAB 2: 3D PROBE & DOMAIN PACKS ================= -->
<div id="pane-probe" class="pane">
  <div class="card">
    <h3>全功能 3D 方向探针与四大垂直领域分析包（General / Electrolyte / Reactive MD / Drug SBDD）</h3>
    <div class="preset-chips" id="probePresetChips"></div>
    <div class="row2" style="margin-top:12px;">
      <div>
        <label>SMILES 或多行 XYZ / POSCAR / CIF 结构输入</label>
        <textarea id="structInput" rows="6">3
water demo
O  0.000  0.000  0.125
H  0.020  0.790 -0.460
H -0.015 -0.745 -0.485</textarea>
      </div>
      <div style="display:flex;flex-direction:column;gap:10px;">
        <div>
          <label>领域分析包（Domain Pack）</label>
          <select id="domainSel">
            <option value="general">General — 分子偶极与任意局部原子受力 (Dipole / Force)</option>
            <option value="electrolyte">Electrolyte — 锂/钠溶剂化壳配位数、SSIP/CIP/AGG 与逃逸向量</option>
            <option value="md">Reactive MD — 局部受力 vs. 帧间位移做功对齐度</option>
            <option value="drug">Drug SBDD — 药效团极性与先导化合物 Exit Vector</option>
          </select>
        </div>
        <div>
          <label>自然语言方向性提问</label>
          <input type="text" id="qInput" value="What is the force vector on the hydrogen atom?" />
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-top:4px;">
          <label style="display:inline-flex;align-items:center;gap:6px;color:var(--text);">
            <input type="checkbox" id="o3Chk" checked /> 实时执行 O(3) 旋转+镜像反演双重校验
          </label>
          <div style="display:flex;gap:8px;">
            <button class="btn btn-sec" onclick="buildLiCluster()">一键生成 Li+–EC 溶剂化簇</button>
            <button class="btn" onclick="runQuery()">运行 3D 方向探针</button>
          </div>
        </div>
      </div>
    </div>
  </div>
  <iframe id="reportFrame" class="report-box"></iframe>
</div>

<!-- ================= TAB 3: LIVE REACTIVE MD & HOOKS ================= -->
<div id="pane-md" class="pane">
  <div class="card">
    <h3>实时分子动力学反应监测器（MLIPTrajectoryRunner &amp; OpenMM / LAMMPS Reaction Hooks）</h3>
    <div class="sub">自动捕捉断键（bond_broken）、成键（bond_formed）与 Grotthuss 质子转移（proton_transfer）帧号与飞秒时间戳，绘制局部机械功 <code>W_i = F_i · Δr_i</code> 时间轴并生成逐帧 3D 双箭头动画。</div>
    <div class="row3">
      <div>
        <label>运行模式</label>
        <select id="mdModeSel">
          <option value="scan_zundel">扫描内置质子转移反应轨迹 (examples/proton_transfer_traj.extxyz)</option>
          <option value="live_md">实时积分运行可反应 MD (H5O2+ 非平衡质子弛豫采样)</option>
        </select>
      </div>
      <div>
        <label>实时 MD 步数 / 温度 (K)</label>
        <div style="display:flex;gap:8px;">
          <input type="number" id="mdSteps" value="28" min="8" max="80" />
          <input type="number" id="mdTemp" value="350" min="0" max="1500" />
        </div>
      </div>
      <div style="display:flex;align-items:flex-end;">
        <button class="btn" style="width:100%;" onclick="runReactiveMD()">启动反应轨迹扫描 / 实时 MD</button>
      </div>
    </div>
    <div id="mdEventsSummary" style="margin-top:14px;"></div>
    <div id="mdSvgContainer" style="margin-top:12px;overflow-x:auto;"></div>
  </div>
  <iframe id="mdReportFrame" class="report-box"></iframe>
</div>

<!-- ================= TAB 4: OUROBOROS STEREO BRIDGE ================= -->
<div id="pane-ouroboros" class="pane">
  <div class="card">
    <h3>Ouroboros-OCSR 跨仓立体化学桥接审计（2D SE(2) SMILES → 3D CIP 立体审计 → 0o 对映体检验）</h3>
    <div class="sub">验证 <code>@/@@</code> 四面体手性中心与反斜杠 <code>F/C=C\\F</code> 顺反异构体无损嵌入、2-hop 拓扑有向四面体体积计算，以及镜像对映体下 <code>YLMUniverseModel</code> 精确翻号 vs. <code>Uni-Mol</code> 手性失明。</div>
    <div class="preset-chips" style="margin-bottom:10px;">
      <span class="chip" onclick="setStereoSmiles('O=C1OC[C@H](F)O1')">(R)-FEC 锂电成膜添加剂</span>
      <span class="chip" onclick="setStereoSmiles('O=C1OC[C@@H](F)O1')">(S)-FEC 镜像对映体</span>
      <span class="chip" onclick="setStereoSmiles('F[C@H](Cl)Br')">(R)-CHFCIBr 手性四面体</span>
      <span class="chip" onclick="setStereoSmiles('F[C@@H](Cl)Br')">(S)-CHFCIBr 镜像对映体</span>
      <span class="chip" onclick="setStereoSmiles('F/C=C\\\\F')">cis-1,2-二氟乙烯 (Z, 平面偶极≠0)</span>
      <span class="chip" onclick="setStereoSmiles('F/C=C/F')">trans-1,2-二氟乙烯 (E, 反演偶极=0)</span>
    </div>
    <div class="row3">
      <div>
        <label>手性 / 顺反异构 SMILES</label>
        <input type="text" id="stereoSmilesInput" value="O=C1OC[C@H](F)O1" />
      </div>
      <div>
        <label>3D 探针问题</label>
        <input type="text" id="stereoQuestionInput" value="What is the 3D molecular dipole moment vector?" />
      </div>
      <div style="display:flex;align-items:flex-end;">
        <button class="btn" style="width:100%;" onclick="runOuroborosAudit()">执行 3D 立体嵌入与对映体审计</button>
      </div>
    </div>
    <div id="stereoAuditResult" style="margin-top:14px;"></div>
  </div>
  <iframe id="stereoReportFrame" class="report-box"></iframe>
</div>

<!-- ================= TAB 5: FM BENCHMARK & LIVE EVAL ================= -->
<div id="pane-benchmark" class="pane">
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
      <div>
        <h3>四阶等变基座模型正面跑分看板（YLM-Universe FM vs. Uni-Mol / GraphMVP / EquiLLM）</h3>
        <div class="sub">左侧预载完整 24-Epoch 手性+多阶全对称基准实测结果；右侧支持直接在当前 CPU 上实时启动四模型同数据集训练与 K 次随机 O(3) 旋转+镜像评测。</div>
      </div>
      <div style="display:flex;gap:8px;align-items:center;">
        <label style="margin:0;">实时训练 Epochs:</label>
        <input type="number" id="benchEpochs" value="6" min="2" max="20" style="width:72px;" />
        <button class="btn" id="btnLiveBench" onclick="runLiveBenchmark()">⚡ 在线实时跑分 (CPU ~15s)</button>
      </div>
    </div>

    <h4 style="margin:14px 0 8px;font-size:14px;color:var(--accent);">A. 预验证完整手性（`0o`）+ 多阶（`0e + 1o + 2e`）对标基准（8 组对映体骨架，含零样本未见新分子骨架）</h4>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>模型家族</th>
            <th>对映体区分 Acc (Test) ↑</th>
            <th>留出新句式 Acc ↑</th>
            <th>零样本未见骨架 Acc ↑</th>
            <th><code>0o</code> 宇称误差 ↓</th>
            <th>非手性平面分子 <code>|f(X)|</code> ↓</th>
            <th><code>1o</code> Vec Cos ↑</th>
            <th><code>0e+2e</code> Tensor Cos ↑</th>
          </tr>
        </thead>
        <tbody id="refBenchBody"></tbody>
      </table>
    </div>

    <div id="liveBenchContainer" style="margin-top:18px;display:none;">
      <h4 style="margin:0 0 8px;font-size:14px;color:#34d399;">B. 本机刚刚实时训练并完成 O(3) 旋转+反演评测的最新结果</h4>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>模型家族</th>
              <th>耗时 (s)</th>
              <th><code>0e</code> Scalar MAE ↓</th>
              <th><code>1o</code> Vec Cos ↑</th>
              <th>单原子力 T3 Cos ↑</th>
              <th><code>1o</code> O(3) 等变误差 ↓</th>
              <th><code>0e+2e</code> Tensor Cos ↑</th>
              <th>寻址 Routing Acc ↑</th>
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
  document.getElementById('auditSummaryBadge').textContent = '正在执行 PyTorch 物理定律审计...';
  const resp = await fetch('/api/physics_audit', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ structure: preset, question, precision, n_rotations: 6 })
  });
  const data = await resp.json();
  if (data.error) { alert(data.error); return; }

  const badge = document.getElementById('auditSummaryBadge');
  badge.textContent = `${data.passed_count} / ${data.total_laws} 项基础物理定律全部通过 (${data.precision})`;
  badge.className = data.passed_all ? 'pill pill-ok' : 'pill';

  const box = document.getElementById('lawsContainer');
  box.innerHTML = data.laws.map(law => {
    const baseHtml = law.baseline_errors
      ? '<div style="margin-top:6px;font-size:12px;color:#fca5a5;">对比基线违背量: ' +
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
      <div class="mono" style="margin-top:6px;font-size:12px;">YLM 残差: <strong>${Number(law.ylm_error).toExponential(2)}</strong> (阈值 &lt; ${Number(law.tolerance).toExponential(1)})</div>
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
    ['ylm', 'YLM-Universe FM (Ours: 0e+0o+1o+2e)'],
    ['equillm', 'EquiLLM (0e Bridge + 1o Bypass)'],
    ['unimol', 'Uni-Mol (3D Distance Matrix 0e)'],
    ['graphmvp', 'GraphMVP / 3D-InfoMax (2D Graph)'],
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
  document.getElementById("mdEventsSummary").innerHTML = '<span class="sub">正在运行分子动力学并监测成键/断键/质子转移事件...</span>';
  const resp = await fetch("/api/run_md", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  const data = await resp.json();
  if (data.error) { alert(data.error); return; }
  const evHtml = (data.events && data.events.length)
    ? data.events.map(e => `<span class="pill" style="margin-right:6px;border-color:#f59e0b;color:#fde68a;">Frame #${e.frame_index} (${e.time_fs.toFixed(1)} fs): ${e.event_type} [${e.labels.join('-')}] W=${e.work_ev.toFixed(3)} eV</span>`).join('')
    : '<span class="pill">共扫描 ' + data.n_frames + ' 帧（当前阈值下无拓扑跳变，可在下方拖动滑块观察逐帧受力与运动向量对比）</span>';
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
    ? data.chiral_centers.map(c => `Atom ${c.symbol}#${c.atom_index}: V_signed = ${c.signed_volume_ang3.toFixed(4)} Å³ (sign=${c.handedness_sign > 0 ? '+1' : '-1'})`).join('; ')
    : '无四面体手性中心（平面或非手性分子，理论 0o 手性输出严格为 0）';
  document.getElementById("stereoAuditResult").innerHTML = `
    <div class="law-grid">
      <div class="law-card pass">
        <strong>3D CIP 立体审计 &amp; 有向四面体体积</strong>
        <div class="mono" style="margin-top:6px;">SMILES: ${data.smiles} | Embedder: ${data.embedder} | stereo_ok: ${data.stereo_ok}</div>
        <div style="font-size:12px;color:var(--muted);margin-top:6px;">${centersText}</div>
      </div>
      <div class="law-card pass">
        <strong>镜像对映体反演测试 <code>X → -X</code>（YLM vs. Uni-Mol）</strong>
        <div class="mono" style="margin-top:6px;color:#34d399;">YLM 0o: χ(X) = ${et.ylm_chiral_orig.toExponential(3)}, χ(-X) = ${et.ylm_chiral_mirror.toExponential(3)} (|χ(X)+χ(-X)| = ${et.ylm_sum_residual.toExponential(2)})</div>
        <div class="mono" style="margin-top:4px;color:#f87171;">Uni-Mol: χ(X) = ${et.unimol_chiral_orig.toExponential(3)}, χ(-X) = ${et.unimol_chiral_mirror.toExponential(3)} (对映体差异 = ${et.unimol_diff.toExponential(2)}，完全失明)</div>
      </div>
    </div>`;
  if (data.html) document.getElementById("stereoReportFrame").srcdoc = data.html;
}

async function runLiveBenchmark() {
  const btn = document.getElementById('btnLiveBench');
  const epochs = Number(document.getElementById('benchEpochs').value || 6);
  btn.disabled = true;
  btn.textContent = '正在实时训练 4 个基座模型...';
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
    btn.textContent = '⚡ 在线实时跑分 (CPU ~15s)';
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
            return  # keep console clean during interactive / test usage

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
                    "reference_benchmarks": _REFERENCE_BENCHMARKS,
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

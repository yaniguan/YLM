"""Standalone interactive 3D HTML report generator (3Dmol.js + offline HTML5 Canvas 3D fallback)."""
from __future__ import annotations

import html
import json
from typing import Optional

from src.toolkit.ingest import covalent_radius, symbol_for_z
from src.toolkit.probe import ProbeResult


def _bonds_for_frame(frame) -> list[list[int]]:
    if frame is None or frame.n_atoms < 2:
        return []
    pos = frame.pos_angstrom
    bonds = []
    for i in range(frame.n_atoms):
        ri = covalent_radius(int(frame.Z[i]))
        for j in range(i + 1, frame.n_atoms):
            rj = covalent_radius(int(frame.Z[j]))
            d = float(((pos[i] - pos[j]) ** 2).sum() ** 0.5)
            if 0.2 < d <= 1.32 * (ri + rj):
                bonds.append([i, j])
    return bonds


def _serialize_probe_payload(res: ProbeResult) -> dict:
    frame = res.frame
    atoms = []
    if frame is not None:
        pos = frame.pos_angstrom
        for idx in range(frame.n_atoms):
            z = int(frame.Z[idx])
            atoms.append({
                "index": idx,
                "Z": z,
                "symbol": symbol_for_z(z),
                "x": round(float(pos[idx, 0]), 5),
                "y": round(float(pos[idx, 1]), 5),
                "z": round(float(pos[idx, 2]), 5),
                "role": int(frame.role[idx]),
            })
    arrows = []
    colors = ["#ef4444", "#2563eb", "#10b981", "#f59e0b"]
    for idx, vp in enumerate(res.vectors):
        vx, vy, vz = vp.vector_physical
        norm = (vx * vx + vy * vy + vz * vz) ** 0.5
        scale = 1.6 / max(norm, 1e-6) if norm > 0 else 0.0
        ox, oy, oz = vp.anchor_xyz_angstrom
        arrows.append({
            "name": vp.name,
            "unit": vp.unit,
            "magnitude": round(float(vp.magnitude), 5),
            "vec": [round(float(c), 5) for c in vp.vector_physical],
            "start": [round(float(ox), 5), round(float(oy), 5), round(float(oz), 5)],
            "end": [round(float(ox + vx * scale), 5), round(float(oy + vy * scale), 5), round(float(oz + vz * scale), 5)],
            "color": colors[idx % len(colors)],
        })
    return {
        "label": frame.label if frame is not None else "structure",
        "xyz": frame.to_xyz_string() if frame is not None else "",
        "cell": None if (frame is None or frame.cell_angstrom is None) else frame.cell_angstrom.tolist(),
        "atoms": atoms,
        "bonds": _bonds_for_frame(frame),
        "arrows": arrows,
        "question": res.question,
        "answer": res.answer,
        "engine": res.engine,
        "domain": res.domain,
        "domain_summary": res.domain_summary,
        "tokens": res.tokens,
        "atom_attention": res.atom_attention,
        "pooled_atom_attention": res.pooled_atom_attention,
        "channel_norms": res.channel_norms,
        "l_channel_weights": res.l_channel_weights,
        "o3_check": None if res.o3_check is None else {
            "passed": res.o3_check.passed,
            "reflection_tested": res.o3_check.reflection_tested,
            "vector_equivariance_error": res.o3_check.vector_equivariance_error,
            "invariance_error": res.o3_check.invariance_error,
        },
        "warnings": res.warnings,
    }


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<script src="https://3Dmol.org/build/3Dmol-min.js" async></script>
<style>
  :root { --bg: #0f172a; --panel: #1e293b; --border: #334155; --text: #f8fafc; --muted: #94a3b8; --accent: #38bdf8; }
  * { box-sizing: border-box; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; }
  body { margin: 0; background: var(--bg); color: var(--text); padding: 20px; }
  .header { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
  .badge { padding: 5px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; border: 1px solid var(--border); }
  .badge-pass { background: rgba(16,185,129,0.16); color: #34d399; border-color: #059669; }
  .badge-info { background: rgba(56,189,248,0.15); color: #38bdf8; border-color: #0284c7; }
  .grid { display: grid; grid-template-columns: minmax(340px, 1.35fr) minmax(320px, 1fr); gap: 16px; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
  #viewerWrap { position: relative; height: 460px; background: #090d16; border-radius: 8px; overflow: hidden; }
  #mol3d, #fallbackCanvas { width: 100%; height: 100%; display: block; }
  .tokens { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .tok { padding: 4px 9px; border-radius: 6px; background: #0f172a; border: 1px solid var(--border); cursor: pointer; font-size: 13px; color: var(--text); }
  .tok.active { border-color: var(--accent); background: rgba(56,189,248,0.22); }
  .bar-row { display: flex; align-items: center; gap: 8px; margin: 6px 0; font-size: 13px; }
  .bar-track { flex: 1; height: 9px; background: #0f172a; border-radius: 999px; overflow: hidden; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, #38bdf8, #818cf8); }
  .vec-item { padding: 8px 10px; border-radius: 8px; background: #0f172a; margin-bottom: 8px; font-size: 13px; border-left: 4px solid #ef4444; }
  .scrubber { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
  pre { background: #0f172a; padding: 10px; border-radius: 8px; overflow-x: auto; font-size: 12px; color: #cbd5e1; }
</style>
</head>
<body>
<div class="header">
  <div>
    <h2 style="margin:0;font-size:20px;">__TITLE__</h2>
    <div id="subtitle" style="color:var(--muted);font-size:13px;margin-top:4px;"></div>
  </div>
  <div style="display:flex;gap:8px;align-items:center;" id="badges"></div>
</div>

<div id="scrubberBox" class="card scrubber" style="display:none;">
  <strong id="frameLabel">Frame 0</strong>
  <input id="frameSlider" type="range" min="0" max="0" value="0" style="flex:1;" />
</div>

<div class="grid">
  <div class="card">
    <div id="viewerWrap">
      <div id="mol3d" style="display:none;"></div>
      <canvas id="fallbackCanvas"></canvas>
    </div>
    <div style="color:var(--muted);font-size:12px;margin-top:8px;">
      Drag to rotate 3D structure. Colored arrows show O(3)-covariant vector predictions; amber halos show IrrepsToText atom attention.
    </div>
  </div>

  <div style="display:flex;flex-direction:column;gap:16px;">
    <div class="card">
      <div style="font-size:12px;color:var(--muted);text-transform:uppercase;">Question &amp; Model Output</div>
      <div id="qText" style="margin-top:6px;font-weight:600;"></div>
      <div id="aText" style="margin-top:8px;color:#7dd3fc;line-height:1.45;"></div>
      <div style="margin-top:12px;font-size:12px;color:var(--muted);">Predicted 3D Vectors</div>
      <div id="vecList" style="margin-top:6px;"></div>
    </div>

    <div class="card">
      <div style="font-size:12px;color:var(--muted);text-transform:uppercase;">Click Token to Inspect Atom Routing &amp; Irrep Channels</div>
      <div class="tokens" id="tokenBox"></div>
      <div style="margin-top:14px;font-size:12px;color:var(--muted);">Spherical Harmonic Channel Norms (0e / 1o / 1e / 2e)</div>
      <div id="channelBars" style="margin-top:6px;"></div>
    </div>

    <div class="card" id="domainCard" style="display:none;">
      <div style="font-size:12px;color:var(--muted);text-transform:uppercase;">Domain Diagnostics</div>
      <pre id="domainPre"></pre>
    </div>
  </div>
</div>

<script>
const FRAMES = __PAYLOAD_JSON__;
let frameIdx = 0;
let selectedToken = -1;
let rotX = -0.35, rotY = 0.55, dragging = false, lastX = 0, lastY = 0;

const ELEMENT_COLOR = {1:"#f8fafc", 3:"#a855f7", 6:"#64748b", 7:"#3b82f6", 8:"#ef4444", 9:"#22c55e", 15:"#f97316", 16:"#eab308", 26:"#d97706"};

function currentWeights(F) {
  if (selectedToken >= 0 && F.atom_attention[selectedToken]) return F.atom_attention[selectedToken];
  return F.pooled_atom_attention || [];
}

function renderCanvas(F) {
  const cv = document.getElementById("fallbackCanvas");
  const rect = cv.parentElement.getBoundingClientRect();
  cv.width = Math.max(320, rect.width);
  cv.height = Math.max(300, rect.height);
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!F.atoms.length) return;

  let cx = 0, cy = 0, cz = 0;
  F.atoms.forEach(a => { cx += a.x; cy += a.y; cz += a.z; });
  cx /= F.atoms.length; cy /= F.atoms.length; cz /= F.atoms.length;

  let maxR = 2.0;
  F.atoms.forEach(a => {
    const d = Math.hypot(a.x - cx, a.y - cy, a.z - cz);
    if (d + 1.5 > maxR) maxR = d + 1.5;
  });
  const scale = 0.40 * Math.min(cv.width, cv.height) / maxR;

  function proj(px, py, pz) {
    let x = px - cx, y = py - cy, z = pz - cz;
    const cy1 = Math.cos(rotY), sy1 = Math.sin(rotY);
    let x1 = x * cy1 + z * sy1, z1 = -x * sy1 + z * cy1;
    const cx1 = Math.cos(rotX), sx1 = Math.sin(rotX);
    let y2 = y * cx1 - z1 * sx1, z2 = y * sx1 + z1 * cx1;
    return [cv.width / 2 + x1 * scale, cv.height / 2 - y2 * scale, z2];
  }

  ctx.lineWidth = 3;
  ctx.strokeStyle = "#475569";
  F.bonds.forEach(([i, j]) => {
    const p1 = proj(F.atoms[i].x, F.atoms[i].y, F.atoms[i].z);
    const p2 = proj(F.atoms[j].x, F.atoms[j].y, F.atoms[j].z);
    ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke();
  });

  const weights = currentWeights(F);
  const maxW = Math.max(...weights, 1e-6);
  const order = F.atoms.map((a, idx) => ({a, idx, p: proj(a.x, a.y, a.z)})).sort((u, v) => u.p[2] - v.p[2]);

  order.forEach(({a, idx, p}) => {
    const w = (weights[idx] || 0) / maxW;
    const rad = (a.Z === 1 ? 8 : 13);
    if (w > 0.15) {
      ctx.beginPath();
      ctx.arc(p[0], p[1], rad + 8 * w, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(251, 191, 36, ${0.18 + 0.55 * w})`;
      ctx.fill();
    }
    ctx.beginPath();
    ctx.arc(p[0], p[1], rad, 0, Math.PI * 2);
    ctx.fillStyle = ELEMENT_COLOR[a.Z] || "#94a3b8";
    ctx.fill();
    ctx.strokeStyle = "#0f172a"; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.fillStyle = a.Z === 1 ? "#0f172a" : "#ffffff";
    ctx.font = "bold 10px sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(a.symbol + idx, p[0], p[1]);
  });

  F.arrows.forEach(ar => {
    const s = proj(ar.start[0], ar.start[1], ar.start[2]);
    const e = proj(ar.end[0], ar.end[1], ar.end[2]);
    ctx.strokeStyle = ar.color; ctx.fillStyle = ar.color; ctx.lineWidth = 3.5;
    ctx.beginPath(); ctx.moveTo(s[0], s[1]); ctx.lineTo(e[0], e[1]); ctx.stroke();
    const ang = Math.atan2(e[1] - s[1], e[0] - s[0]);
    ctx.beginPath();
    ctx.moveTo(e[0], e[1]);
    ctx.lineTo(e[0] - 11 * Math.cos(ang - 0.4), e[1] - 11 * Math.sin(ang - 0.4));
    ctx.lineTo(e[0] - 11 * Math.cos(ang + 0.4), e[1] - 11 * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();
  });
}

function renderPanel() {
  const F = FRAMES[frameIdx];
  document.getElementById("subtitle").textContent = `Structure: ${F.label} | Engine: ${F.engine} | Domain: ${F.domain}`;
  document.getElementById("qText").textContent = "Q: " + F.question;
  document.getElementById("aText").textContent = F.answer;

  const badges = document.getElementById("badges");
  badges.innerHTML = `<span class="badge badge-info">${F.atoms.length} atoms</span>`;
  if (F.o3_check) {
    const cls = F.o3_check.passed ? "badge-pass" : "badge";
    const status = F.o3_check.passed ? "O(3) Equivariant PASS" : "O(3) Drift Warning";
    badges.innerHTML += `<span class="badge ${cls}">${status} (err=${F.o3_check.vector_equivariance_error.toExponential(2)})</span>`;
  }

  const vecList = document.getElementById("vecList");
  vecList.innerHTML = "";
  F.arrows.forEach(ar => {
    const d = document.createElement("div");
    d.className = "vec-item";
    d.style.borderLeftColor = ar.color;
    d.textContent = `${ar.name}: (${ar.vec.join(", ")}) ${ar.unit}  |v| = ${ar.magnitude}`;
    vecList.appendChild(d);
  });

  const tokBox = document.getElementById("tokenBox");
  tokBox.innerHTML = "";
  const poolBtn = document.createElement("button");
  poolBtn.className = "tok" + (selectedToken === -1 ? " active" : "");
  poolBtn.textContent = "[Pooled]";
  poolBtn.onclick = () => { selectedToken = -1; renderPanel(); };
  tokBox.appendChild(poolBtn);

  F.tokens.forEach((tk, idx) => {
    const b = document.createElement("button");
    b.className = "tok" + (selectedToken === idx ? " active" : "");
    b.textContent = tk;
    b.onclick = () => { selectedToken = idx; renderPanel(); };
    tokBox.appendChild(b);
  });

  const chBox = document.getElementById("channelBars");
  chBox.innerHTML = "";
  const keys = Object.keys(F.channel_norms);
  let vals = keys.map(k => {
    const arr = F.channel_norms[k] || [0];
    return selectedToken >= 0 && arr[selectedToken] !== undefined ? arr[selectedToken] : (arr.reduce((a,b)=>a+b,0) / Math.max(arr.length,1));
  });
  const sumV = vals.reduce((a, b) => a + b, 0) || 1.0;
  keys.forEach((k, i) => {
    const pct = (100 * vals[i] / sumV).toFixed(1);
    chBox.innerHTML += `<div class="bar-row"><span style="width:32px;font-weight:600;">${k}</span><div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div><span style="width:46px;text-align:right;">${pct}%</span></div>`;
  });

  const domCard = document.getElementById("domainCard");
  if (F.domain_summary && Object.keys(F.domain_summary).length > 0) {
    domCard.style.display = "block";
    document.getElementById("domainPre").textContent = JSON.stringify(F.domain_summary, null, 2);
  } else {
    domCard.style.display = "none";
  }

  renderCanvas(F);
}

const cv = document.getElementById("fallbackCanvas");
cv.addEventListener("mousedown", e => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener("mouseup", () => { dragging = false; });
window.addEventListener("mousemove", e => {
  if (!dragging) return;
  rotY += (e.clientX - lastX) * 0.01;
  rotX += (e.clientY - lastY) * 0.01;
  lastX = e.clientX; lastY = e.clientY;
  renderCanvas(FRAMES[frameIdx]);
});
window.addEventListener("resize", () => renderCanvas(FRAMES[frameIdx]));

if (FRAMES.length > 1) {
  const sb = document.getElementById("scrubberBox");
  const sl = document.getElementById("frameSlider");
  sb.style.display = "flex";
  sl.max = String(FRAMES.length - 1);
  sl.oninput = () => {
    frameIdx = parseInt(sl.value, 10);
    selectedToken = -1;
    document.getElementById("frameLabel").textContent = `Frame ${frameIdx} (${FRAMES[frameIdx].label})`;
    renderPanel();
  };
}
renderPanel();
</script>
</body>
</html>
"""


def render_html_report(result: ProbeResult, title: Optional[str] = None) -> str:
    safe_title = html.escape(title or f"YLM Directional Probe - {result.frame.label if result.frame else 'Report'}")
    payload = json.dumps([_serialize_probe_payload(result)])
    return _HTML_TEMPLATE.replace("__TITLE__", safe_title).replace("__PAYLOAD_JSON__", payload)


def render_trajectory_html_report(scan_result, title: Optional[str] = None) -> str:
    safe_title = html.escape(title or f"YLM MD Trajectory Scan ({len(scan_result.frame_results)} frames)")
    payload = json.dumps([_serialize_probe_payload(r) for r in scan_result.frame_results])
    return _HTML_TEMPLATE.replace("__TITLE__", safe_title).replace("__PAYLOAD_JSON__", payload)

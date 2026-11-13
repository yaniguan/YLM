"""Standalone interactive 3D HTML report generator."""
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
    colors = ["#141413", "#1d4ed8", "#15803d", "#b45309"]
    for idx, vp in enumerate(res.vectors):
        vx, vy, vz = vp.vector_physical
        norm = (vx * vx + vy * vy + vz * vz) ** 0.5
        scale = 1.65 / max(norm, 1e-6) if norm > 0 else 0.0
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
<style>
  :root {
    --bg: #faf9f7;
    --surface: #ffffff;
    --subtle: #f3f2ee;
    --border: #e8e6e0;
    --ink: #141413;
    --ink-2: #5e5d59;
    --ink-3: #8e8c85;
    --ok-fg: #166534;
    --ok-bg: #eef8f1;
    --ok-bd: #cbead5;
  }
  [data-theme="dark"] {
    --bg: #111110;
    --surface: #191918;
    --subtle: #222220;
    --border: #2c2c29;
    --ink: #ededec;
    --ink-2: #a1a09a;
    --ink-3: #6f6e69;
    --ok-fg: #4ade80;
    --ok-bg: rgba(34,197,94,0.10);
    --ok-bd: rgba(34,197,94,0.25);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    padding: 28px;
    -webkit-font-smoothing: antialiased;
  }
  body.embedded {
    background: transparent;
    padding: 0;
  }
  .wrap { max-width: 1080px; margin: 0 auto; }
  .header {
    display: flex; justify-content: space-between; align-items: center;
    gap: 12px; flex-wrap: wrap; margin-bottom: 20px;
  }
  body.embedded .standalone-title { display: none; }
  .badge {
    padding: 4px 11px; border-radius: 999px; font-size: 12px; font-weight: 500;
    border: 1px solid var(--border); background: var(--surface); color: var(--ink-2);
  }
  .badge-pass { background: var(--ok-bg); color: var(--ok-fg); border-color: var(--ok-bd); }
  .grid {
    display: grid; grid-template-columns: minmax(340px, 1.25fr) minmax(300px, 1fr);
    gap: 20px; align-items: start;
  }
  @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 20px;
  }
  #viewerWrap {
    position: relative; height: 420px; background: var(--subtle);
    border-radius: 12px; overflow: hidden; border: 1px solid var(--border);
  }
  #fallbackCanvas { width: 100%; height: 100%; display: block; cursor: grab; }
  #fallbackCanvas:active { cursor: grabbing; }
  .section-label { font-size: 12px; font-weight: 500; color: var(--ink-3); margin-bottom: 8px; }
  .tokens { display: flex; flex-wrap: wrap; gap: 6px; }
  .tok {
    padding: 5px 11px; border-radius: 999px; background: var(--subtle);
    border: 1px solid transparent; cursor: pointer; font-size: 12.5px;
    color: var(--ink-2); transition: all 0.15s ease;
  }
  .tok:hover { color: var(--ink); border-color: var(--border); }
  .tok.active { background: var(--ink); color: var(--surface); font-weight: 500; }
  .bar-row { display: flex; align-items: center; gap: 10px; margin: 7px 0; font-size: 12.5px; color: var(--ink-2); }
  .bar-track { flex: 1; height: 6px; background: var(--subtle); border-radius: 999px; overflow: hidden; }
  .bar-fill { height: 100%; background: var(--ink); border-radius: 999px; }
  .vec-item {
    display: flex; justify-content: space-between; align-items: center; gap: 12px;
    padding: 10px 12px; border-radius: 10px; background: var(--subtle);
    margin-bottom: 8px; font-size: 12.5px;
  }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
  .scrubber { display: flex; align-items: center; gap: 14px; margin-bottom: 16px; padding: 14px 18px; }
  input[type="range"] { flex: 1; accent-color: var(--ink); }
  details summary { cursor: pointer; font-size: 12.5px; color: var(--ink-2); font-weight: 500; user-select: none; }
  pre {
    background: var(--subtle); padding: 12px; border-radius: 10px;
    overflow-x: auto; font-size: 12px; color: var(--ink-2); margin: 10px 0 0;
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="header standalone-title">
    <div>
      <h2 style="margin:0;font-size:19px;font-weight:600;letter-spacing:-0.02em;">__TITLE__</h2>
      <div id="subtitle" style="color:var(--ink-3);font-size:13px;margin-top:3px;"></div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;" id="badgesTop"></div>
  </div>

  <div id="scrubberBox" class="card scrubber" style="display:none;">
    <span id="frameLabel" style="font-size:13px;font-weight:500;min-width:110px;">Frame 0</span>
    <input id="frameSlider" type="range" min="0" max="0" value="0" />
  </div>

  <div class="grid">
    <div class="card" style="padding:14px;">
      <div id="viewerWrap">
        <canvas id="fallbackCanvas"></canvas>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 4px 2px;font-size:12px;color:var(--ink-3);">
        <span>Drag to orbit · Amber glow shows attended atoms</span>
        <span id="atomCountLabel"></span>
      </div>
    </div>

    <div style="display:flex;flex-direction:column;gap:14px;">
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px;">
          <span class="section-label" style="margin:0;">Response</span>
          <div id="badgesInline" style="display:flex;gap:6px;flex-wrap:wrap;"></div>
        </div>
        <div id="qText" style="font-size:14.5px;font-weight:500;line-height:1.45;"></div>
        <div id="aText" style="margin-top:8px;font-size:13.5px;color:var(--ink-2);line-height:1.6;"></div>
        <div id="vecWrap" style="margin-top:16px;">
          <div class="section-label">3D Vectors</div>
          <div id="vecList"></div>
        </div>
      </div>

      <div class="card">
        <div class="section-label">Word-to-Atom Routing</div>
        <div class="tokens" id="tokenBox"></div>
        <div class="section-label" style="margin-top:18px;">Symmetry Harmonics</div>
        <div id="channelBars"></div>
      </div>

      <div class="card" id="domainCard" style="display:none;padding:14px 18px;">
        <details>
          <summary>Raw diagnostics</summary>
          <pre id="domainPre" class="mono"></pre>
        </details>
      </div>
    </div>
  </div>
</div>

<script>
if (window.self !== window.top) document.body.classList.add("embedded");
window.addEventListener("message", e => {
  if (e.data && e.data.type === "ylm-theme") {
    document.documentElement.setAttribute("data-theme", e.data.theme);
    renderCanvas(FRAMES[frameIdx]);
  }
});

const FRAMES = __PAYLOAD_JSON__;
let frameIdx = 0;
let selectedToken = -1;
let rotX = -0.35, rotY = 0.55, dragging = false, lastX = 0, lastY = 0;

const ELEMENT_COLOR = {
  1: "#ffffff", 3: "#8b5cf6", 6: "#3a3a38", 7: "#2563eb",
  8: "#dc2626", 9: "#16a34a", 15: "#ea580c", 16: "#ca8a04",
  17: "#0d9488", 26: "#b45309", 35: "#be185d"
};

function notifyHeight() {
  if (window.self !== window.top) {
    const h = document.documentElement.scrollHeight;
    window.parent.postMessage({ type: "ylm-frame-height", height: h }, "*");
  }
}

function currentWeights(F) {
  if (selectedToken >= 0 && F.atom_attention[selectedToken]) return F.atom_attention[selectedToken];
  return F.pooled_atom_attention || [];
}

function renderCanvas(F) {
  const cv = document.getElementById("fallbackCanvas");
  const rect = cv.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(320, rect.width || 440);
  const h = Math.max(300, rect.height || 420);
  cv.width = w * dpr;
  cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  if (!F.atoms.length) return;

  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  let cx = 0, cy = 0, cz = 0;
  F.atoms.forEach(a => { cx += a.x; cy += a.y; cz += a.z; });
  cx /= F.atoms.length; cy /= F.atoms.length; cz /= F.atoms.length;

  let maxR = 2.1;
  F.atoms.forEach(a => {
    const d = Math.hypot(a.x - cx, a.y - cy, a.z - cz);
    if (d + 1.5 > maxR) maxR = d + 1.5;
  });
  const scale = 0.40 * Math.min(w, h) / maxR;

  function proj(px, py, pz) {
    const x = px - cx, y = py - cy, z = pz - cz;
    const cy1 = Math.cos(rotY), sy1 = Math.sin(rotY);
    const x1 = x * cy1 + z * sy1, z1 = -x * sy1 + z * cy1;
    const cx1 = Math.cos(rotX), sx1 = Math.sin(rotX);
    const y2 = y * cx1 - z1 * sx1, z2 = y * sx1 + z1 * cx1;
    return [w / 2 + x1 * scale, h / 2 - y2 * scale, z2];
  }

  ctx.lineWidth = 2.6;
  ctx.lineCap = "round";
  ctx.strokeStyle = isDark ? "#52524e" : "#c8c5bc";
  F.bonds.forEach(([i, j]) => {
    const p1 = proj(F.atoms[i].x, F.atoms[i].y, F.atoms[i].z);
    const p2 = proj(F.atoms[j].x, F.atoms[j].y, F.atoms[j].z);
    ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke();
  });

  const weights = currentWeights(F);
  const maxW = Math.max(...weights, 1e-6);
  const meanW = weights.length ? weights.reduce((a, b) => a + b, 0) / weights.length : maxW;
  const hasPeakedAttn = maxW > 1.25 * meanW;
  const order = F.atoms.map((a, idx) => ({a, idx, p: proj(a.x, a.y, a.z)})).sort((u, v) => u.p[2] - v.p[2]);

  order.forEach(({a, idx, p}) => {
    const wt = (weights[idx] || 0) / maxW;
    const rad = a.Z === 1 ? 8 : 13;
    if (hasPeakedAttn && wt > 0.5) {
      ctx.beginPath();
      ctx.arc(p[0], p[1], rad + 7 * wt, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(217, 119, 6, ${0.14 + 0.24 * wt})`;
      ctx.fill();
    }
    const baseCol = ELEMENT_COLOR[a.Z] || "#78716c";
    ctx.beginPath();
    ctx.arc(p[0], p[1], rad, 0, Math.PI * 2);
    ctx.fillStyle = baseCol;
    ctx.fill();
    const spec = ctx.createRadialGradient(p[0] - rad * 0.32, p[1] - rad * 0.32, rad * 0.05, p[0] - rad * 0.15, p[1] - rad * 0.15, rad * 0.75);
    spec.addColorStop(0, "rgba(255,255,255,0.42)");
    spec.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = spec;
    ctx.fill();
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.18)" : "rgba(20,20,19,0.16)";
    ctx.lineWidth = 1;
    ctx.stroke();

    if (a.Z !== 1) {
      ctx.fillStyle = "#ffffff";
      ctx.font = "600 10px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(a.symbol, p[0], p[1]);
    }
  });

  F.arrows.forEach((ar, idx) => {
    const s = proj(ar.start[0], ar.start[1], ar.start[2]);
    const e = proj(ar.end[0], ar.end[1], ar.end[2]);
    const col = (isDark && ar.color === "#141413") ? "#ededec" : ar.color;
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2.6;
    ctx.beginPath(); ctx.moveTo(s[0], s[1]); ctx.lineTo(e[0], e[1]); ctx.stroke();
    const ang = Math.atan2(e[1] - s[1], e[0] - s[0]);
    ctx.beginPath();
    ctx.moveTo(e[0], e[1]);
    ctx.lineTo(e[0] - 9.5 * Math.cos(ang - 0.38), e[1] - 9.5 * Math.sin(ang - 0.38));
    ctx.lineTo(e[0] - 9.5 * Math.cos(ang + 0.38), e[1] - 9.5 * Math.sin(ang + 0.38));
    ctx.closePath(); ctx.fill();
  });
}

function renderPanel() {
  const F = FRAMES[frameIdx];
  document.getElementById("subtitle").textContent = `${F.label} · ${F.domain}`;
  document.getElementById("atomCountLabel").textContent = `${F.atoms.length} atoms`;
  document.getElementById("qText").textContent = F.question;
  document.getElementById("aText").textContent = F.answer;

  let badgeHtml = "";
  if (F.o3_check) {
    const cls = F.o3_check.passed ? "badge badge-pass" : "badge";
    const status = F.o3_check.passed ? "O(3) Equivariant PASS" : "O(3) Drift";
    badgeHtml = `<span class="${cls}">${status} · ${F.o3_check.vector_equivariance_error.toExponential(1)}</span>`;
  }
  document.getElementById("badgesTop").innerHTML = badgeHtml;
  document.getElementById("badgesInline").innerHTML = badgeHtml;

  const vecList = document.getElementById("vecList");
  vecList.innerHTML = "";
  document.getElementById("vecWrap").style.display = F.arrows.length ? "block" : "none";
  F.arrows.forEach(ar => {
    const d = document.createElement("div");
    d.className = "vec-item";
    d.innerHTML = `<span style="display:inline-flex;align-items:center;gap:7px;font-weight:500;">
        <span style="width:8px;height:8px;border-radius:99px;background:${ar.color};display:inline-block;"></span>${ar.name}
      </span>
      <span class="mono" style="color:var(--ink-2);">(${ar.vec.map(v => v.toFixed(3)).join(", ")}) ${ar.unit}</span>`;
    vecList.appendChild(d);
  });

  const tokBox = document.getElementById("tokenBox");
  tokBox.innerHTML = "";
  const poolBtn = document.createElement("button");
  poolBtn.className = "tok" + (selectedToken === -1 ? " active" : "");
  poolBtn.textContent = "All words";
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
  const vals = keys.map(k => {
    const arr = F.channel_norms[k] || [0];
    return selectedToken >= 0 && arr[selectedToken] !== undefined
      ? arr[selectedToken]
      : (arr.reduce((a, b) => a + b, 0) / Math.max(arr.length, 1));
  });
  const sumV = vals.reduce((a, b) => a + b, 0) || 1.0;
  keys.forEach((k, i) => {
    const pct = (100 * vals[i] / sumV).toFixed(1);
    chBox.innerHTML += `<div class="bar-row">
      <span class="mono" style="width:28px;color:var(--ink);font-weight:500;">${k}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
      <span class="mono" style="width:44px;text-align:right;">${pct}%</span>
    </div>`;
  });

  const domCard = document.getElementById("domainCard");
  if (F.domain_summary && Object.keys(F.domain_summary).length > 0) {
    domCard.style.display = "block";
    document.getElementById("domainPre").textContent = JSON.stringify(F.domain_summary, null, 2);
  } else {
    domCard.style.display = "none";
  }

  renderCanvas(F);
  setTimeout(notifyHeight, 30);
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
window.addEventListener("resize", () => { renderCanvas(FRAMES[frameIdx]); notifyHeight(); });

if (FRAMES.length > 1) {
  const sb = document.getElementById("scrubberBox");
  const sl = document.getElementById("frameSlider");
  sb.style.display = "flex";
  sl.max = String(FRAMES.length - 1);
  sl.oninput = () => {
    frameIdx = parseInt(sl.value, 10);
    selectedToken = -1;
    document.getElementById("frameLabel").textContent = `Frame ${frameIdx} / ${FRAMES.length - 1}`;
    renderPanel();
  };
}
renderPanel();
</script>
</body>
</html>
"""


def render_html_report(result: ProbeResult, title: Optional[str] = None) -> str:
    safe_title = html.escape(title or f"YLM Probe · {result.frame.label if result.frame else 'Report'}")
    payload = json.dumps([_serialize_probe_payload(result)])
    return _HTML_TEMPLATE.replace("__TITLE__", safe_title).replace("__PAYLOAD_JSON__", payload)


def render_trajectory_html_report(scan_result, title: Optional[str] = None) -> str:
    safe_title = html.escape(title or f"YLM Trajectory ({len(scan_result.frame_results)} frames)")
    payload = json.dumps([_serialize_probe_payload(r) for r in scan_result.frame_results])
    return _HTML_TEMPLATE.replace("__TITLE__", safe_title).replace("__PAYLOAD_JSON__", payload)

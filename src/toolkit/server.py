"""Zero-extra-dependency interactive Web Studio server (`ylm serve`) built on stdlib http.server."""
from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Optional

from src.toolkit.ingest import build_solvation_cluster, load_structure
from src.toolkit.probe import YLMProbe
from src.toolkit.viz import render_html_report

_STUDIO_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>YLM Directional Chemistry Studio</title>
<style>
  body { margin: 0; background: #0f172a; color: #f8fafc; font-family: ui-sans-serif, system-ui, sans-serif; padding: 20px; }
  .bar { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; background: #1e293b; padding: 16px; border-radius: 12px; border: 1px solid #334155; }
  textarea, input, select { width: 100%; background: #0f172a; color: #f8fafc; border: 1px solid #475569; border-radius: 8px; padding: 8px 10px; font-size: 13px; box-sizing: border-box; }
  button { background: #0284c7; color: white; border: none; padding: 10px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; }
  button:hover { background: #0369a1; }
  iframe { width: 100%; height: 680px; border: 1px solid #334155; border-radius: 12px; margin-top: 16px; background: #0f172a; }
</style>
</head>
<body>
<h2 style="margin-top:0;">YLM Directional Chemistry Studio</h2>
<div class="bar">
  <div>
    <label style="font-size:12px;color:#94a3b8;">SMILES or Multi-line XYZ / POSCAR Structure</label>
    <textarea id="structInput" rows="5">3
water demo
O  0.000  0.000  0.125
H  0.020  0.790 -0.460
H -0.015 -0.745 -0.485</textarea>
  </div>
  <div style="display:flex;flex-direction:column;gap:8px;">
    <div>
      <label style="font-size:12px;color:#94a3b8;">Domain Pack</label>
      <select id="domainSel">
        <option value="general">General (Dipole / Force Probe)</option>
        <option value="electrolyte">Electrolyte (Solvation Shell &amp; Ion Escape)</option>
        <option value="md">MD Frame (Force vs Motion Alignment)</option>
        <option value="drug">Drug Lead-Opt (Pharmacophore &amp; Exit Vector)</option>
      </select>
    </div>
    <div>
      <label style="font-size:12px;color:#94a3b8;">Natural-Language Directional Question</label>
      <input id="qInput" value="What is the force vector on the hydrogen atom?" />
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
      <label style="font-size:13px;"><input type="checkbox" id="o3Chk" checked style="width:auto;" /> Live O(3) rotation+reflection check</label>
      <button onclick="runQuery()">Probe 3D Direction</button>
    </div>
  </div>
</div>
<iframe id="reportFrame"></iframe>
<script>
async function runQuery() {
  const payload = {
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
runQuery();
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
                })
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/ask":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                if body.get("solvation_recipe"):
                    rec = body["solvation_recipe"]
                    frame = build_solvation_cluster(
                        solvent_smiles=rec.get("solvents", ["O=C1OCCO1", "O=C1OCCO1"]),
                        ion_symbol=rec.get("ion", "Li"),
                    )
                else:
                    frame = load_structure(body.get("structure", "O"))
                question = str(body.get("question") or "What is the dipole vector?")
                domain = str(body.get("domain") or "general")
                verify = bool(body.get("verify_o3", True))
                res = active_probe.ask(frame, question, domain=domain, verify_o3=verify)
                payload = res.to_dict()
                payload["html"] = render_html_report(res)
                self._send_json(HTTPStatus.OK, payload)
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return ThreadingHTTPServer((host, port), StudioHandler)

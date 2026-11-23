#!/bin/bash
# Build colab_bundle/ylm_data.zip = the exact data files the v1/pol/quad runs use (they are git-ignored).
# Upload it to Google Drive (MyDrive/ylm/); colab/run_packs.ipynb clones the code from GitHub.
set -e
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON:-$(command -v python3 || command -v python)}"
rm -rf colab_bundle && mkdir -p colab_bundle
"$PYTHON_BIN" - <<'PY'
import os
import sys
import zipfile

names = [
    "qm9_xtb.pkl",
    "tensorqa_v1.json",
    "parsed_v1.json",
    "qm7b_pol.pkl",
    "tensorqa_pol.json",
    "parsed_pol.json",
    "qm9_quad.pkl",
    "tensorqa_quad.json",
    "text_quad_matscibert.pt",
]
present = [n for n in names if os.path.exists(os.path.join("data", n))]
missing = [n for n in names if n not in present]
if not present:
    print("error: none of the expected files exist under data/", file=sys.stderr)
    sys.exit(1)
with zipfile.ZipFile("colab_bundle/ylm_data.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for n in present:
        z.write("data/" + n, "data/" + n)
print(f"packed {len(present)}/{len(names)} data files into colab_bundle/ylm_data.zip")
if missing:
    print("skipped absent optional files:", ", ".join(missing))
PY
ls -la colab_bundle/ylm_data.zip

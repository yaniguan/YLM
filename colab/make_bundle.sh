#!/bin/bash
# Build colab_bundle/ylm_data.zip = the exact data files the v1/pol/quad runs use (they are git-ignored).
# Upload it to Google Drive (MyDrive/ylm/); colab/run_packs.ipynb clones the code from GitHub.
set -e
cd "$(git rev-parse --show-toplevel)"
rm -rf colab_bundle && mkdir -p colab_bundle
python3 - <<'PY'
import zipfile, os
names = ["qm9_xtb.pkl","tensorqa_v1.json","parsed_v1.json","qm7b_pol.pkl","tensorqa_pol.json","parsed_pol.json","tensorqa_quad.json","text_quad_matscibert.pt"]
with zipfile.ZipFile("colab_bundle/ylm_data.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for n in names: z.write("data/" + n, "data/" + n)
PY
ls -la colab_bundle/ylm_data.zip

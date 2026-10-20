#!/bin/bash
# Build colab_bundle/ylm_bundle.zip = tracked code at HEAD + the exact data files the v1/pol/quad runs use.
# Upload the zip to Google Drive (MyDrive/ylm/) and open colab/run_packs.ipynb.
set -e
cd "$(git rev-parse --show-toplevel)"
rm -rf colab_bundle && mkdir -p colab_bundle/stage
git archive HEAD | tar -x -C colab_bundle/stage
mkdir -p colab_bundle/stage/data
for f in qm9_xtb.pkl tensorqa_v1.json parsed_v1.json qm7b_pol.pkl tensorqa_pol.json parsed_pol.json tensorqa_quad.json text_quad_matscibert.pt; do
  cp data/$f colab_bundle/stage/data/
done
(cd colab_bundle/stage && zip -qr ../ylm_bundle.zip .)
ls -la colab_bundle/ylm_bundle.zip

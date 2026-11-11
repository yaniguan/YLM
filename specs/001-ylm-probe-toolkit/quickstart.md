# Quickstart: YLM Probe & Domain Toolkit

## 1. Editable Install

```bash
pip install -e .
```

## 2. Instant CLI Directional Probe (Zero Training)

```bash
# Probe a molecule (.xyz or SMILES), verify O(3) equivariance live, and export a 3D HTML report
ylm ask examples/water.xyz "What is the force vector on the hydrogen atom?" --verify-o3 --html water_probe.html

# Electrolyte Li+ solvation shell diagnostics + escape vector
ylm solvation examples/li_ec_shell.xyz --html solvation_report.html

# Scan a multi-frame MD trajectory (.extxyz) with 1o inter-frame displacement analysis
ylm scan-traj examples/proton_transfer_traj.extxyz --query "What is the restoring force on the transferring hydrogen?" --html md_report.html

# Drug ligand + pocket pharmacophore & steric-safe 3D exit vectors
ylm drug examples/indole_pocket.xyz --html drug_exit_vectors.html
```

## 3. Python SDK (3 Lines)

```python
from src.toolkit import YLMProbe

probe = YLMProbe.default()  # loads checkpoints/v0/ylm_matscibert_s0.pt
res = probe.ask("examples/water.xyz", "Which way does the dipole point?", verify_o3=True)
res.to_html("water.html")
```

## 4. Lightweight `IrrepsAdapter` + `<vec>` Post-Training

```bash
ylm posttrain --input examples/proton_transfer_traj.extxyz --out runs/my_directional_lm --steps 30 --dpo-steps 5
ylm ask examples/water.xyz "Q: Force on O(2H)? A: " --ckpt runs/my_directional_lm --verify-o3
```

## 5. Launch Interactive Local Web Studio & Export Hugging Face Benchmark

```bash
ylm serve --port 8765
# Open http://127.0.0.1:8765 in any browser

# Package TensorQA-v0 (+ optional T4/T5 compositional & l=2 quadrupole) for Hugging Face Datasets
ylm export-hf --labels data/qm9_xtb.pkl --v0 data/tensorqa_v0.json --out dist/hf_tensorqa --anonymized
```

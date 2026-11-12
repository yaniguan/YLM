# `waiting4gpu.md` — GPU Training Guide

**Current status**: All core code, 3D symmetry layers, crystal repeat boxes (PBC), 3x3 matrix heads, MACE / NequIP adapters, Ouroboros 2D-to-3D stereo bridge, OpenMM / LAMMPS reaction hooks, and the 5-tab browser Studio already run and pass tests on CPU. Once a GPU is ready, follow the steps below to train at full scale.

---

## 1. What Works on CPU Right Now vs. What Needs a GPU

| Task | Runs on CPU? | Notes |
| :--- | :---: | :--- |
| **`ylm` CLI, 3D HTML reports, and browser Studio (`ylm serve`)** | **Yes** | Uses the small default checkpoint in `checkpoints/v0`. |
| **Testing `MACE-MP-0-small` or `MACE-OFF23-small` on 1 molecule or a short trajectory** | **Yes** | Fast enough to check forces and reaction timelines. |
| **Data prep and small smoke tests (100–600 rows)** | **Yes (do this first)** | Catches path or data-format bugs before spending GPU hours. |
| **Million-molecule DFT training vs. Uni-Mol / GraphMVP** | **Needs GPU** | Too slow on CPU. |
| **Long reactive MD runs (20–100 ps with MACE / NequIP)** | **Needs GPU** | Tens of thousands of force steps need a single GPU. |

---

## 2. Two Main Goals

### Goal A: A 3D Chemistry Model That Keeps Directions and Handedness (`YLMUniverseModel`)
- **Why existing models fall short**:
  - `GraphMVP` and `3D-InfoMax` only use 2D bond graphs at test time, so they have no 3D directions at all.
  - `Uni-Mol` only reads atom-to-atom distances (`d_ij`). Distances do not change when you spin a molecule or flip it in a mirror, so Uni-Mol cannot rotate force/dipole arrows and cannot tell left-handed `(R)` from right-handed `(S)` molecules.
  - `EquiLLM` squashes features into plain numbers before talking to the text model, so it loses handedness and struggles to combine 3D vectors across multiple atoms.
- **How YLM solves this**:
  - Plugs into pretrained 3D atom models (`MACE-MP-0` for 89 elements in crystals/materials, `MACE-OFF23` for organic/drug molecules, or `NequIP`/`Allegro`).
  - Uses `IrrepsToText` cross-attention so words can pick specific atoms while keeping 3D arrows, 3x3 matrices, and left/right handedness intact.
  - Predicts **plain numbers** (energy, gap, solubility), **handedness** (`+` vs. `-` in a mirror, `0` on flat shapes), **3D arrows** (dipole, atom force, force differences), and **symmetric 3x3 matrices** (polarizability, quadrupole, crystal stress) in one model.

### Goal B: Live Reaction Monitor for MD Simulations (`OpenMM` / `LAMMPS`)
- Attach `OpenMMReactionReporter` or `LAMMPSReactionHook` (`src/toolkit/md_hooks.py`) while running reactive MD.
- Tracks smooth bond numbers between atoms with two thresholds (so normal heat vibration does not trigger false alarms).
- Whenever a **bond forms**, **bond breaks**, or a **proton (`H+`) jumps**, it records the frame number, time in femtoseconds, and work `W = Force · Step`, then saves a JSON log, an SVG/PNG timeline chart, and a 3D HTML viewer.

---

## 3. Ready-to-Use Code in This Repo

| File | What it does |
| :--- | :--- |
| `src/mace_backbone.py` (`MACEEncoder`) | Loads `mace-mp-0-{small,medium,large}` or `mace-off23-{small,medium,large}` with unit conversion, crystal boxes, and exact `F = -dE/dx` forces. |
| `src/nequip_backbone.py` (`NequIPEncoder`) | Loads compiled TorchScript `nequip-deploy` / `allegro` `.pth` files without needing the `nequip` package installed. |
| `src/toolkit/mlip_md.py` (`MLIPTrajectoryRunner`, `ylm run-md`) | Runs reactive MD steps with MACE, NequIP, OpenMM, or the built-in test potential and saves reaction plots + 3D HTML reports. |
| `src/toolkit/ouroboros_bridge.py` (`ylm from-ouroboros`) | Connects `yaniguan/ouroboros-ocsr` 2D drawings, stereo SMILES, `predictions.csv`, and `.extxyz` bundles to 3D `YLM` with `R/S` and `cis/trans` checks. |
| `src/universe_fm.py` (`YLMUniverseModel`) | Multi-task model predicting numbers, handedness, 3D arrows, and 3x3 matrices (`scripts/30_train_universe_fm.py`). |
| `src/universe_data.py` | Data loader and question generator with crystal neighbor search, signed chiral volume, and 35+ question templates (`scripts/31_prepare_universe_data.py`). |
| `src/fm_baselines.py` | Side-by-side baselines: `UniMolStyle3DBaseline`, `GraphMVP3DInfoMaxBaseline`, and `EquiLLMMultiTaskBaseline` (`scripts/32_benchmark_vs_unimol.py`, `scripts/35_build_and_eval_chiral_benchmark.py`). |
| `src/toolkit/physics_audit.py` & `server.py` | Checks 8 physics rules (`ylm audit-physics`) and serves the 5-tab browser Studio (`ylm serve`). |

---

## 4. CPU Smoke Tests

Run these two commands on CPU before using GPU time. They train tiny models on a small built-in dataset:

```bash
python scripts/32_benchmark_vs_unimol.py --tiny --epochs 20
python scripts/35_build_and_eval_chiral_benchmark.py --tiny --epochs 24
```

Results from one CPU run on Nov 12, 2026 are saved in [`results/cpu_smoke_nov12/`](results/cpu_smoke_nov12/README.md). The tables below are copied from those files. These are tiny smoke tests, not final results.

### 4.1 Multi-task 3D comparison (`unimol_benchmark.json`, test split)

| Model | Energy MAE ↓ | Atom Pick Acc ↑ | 3D Arrow Cos ↑ | Rotation Error ↓ | 3x3 Matrix Cos ↑ | New Wording Arrow Cos ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| GraphMVP / 3D-InfoMax | 0.1678 | 0.750 | +0.0351 | 1.31e+00 | +0.6500 | +0.0551 |
| Uni-Mol (3D Dist-Transformer) | 0.1895 | 0.764 | +0.0661 | 1.31e+00 | +0.6586 | +0.0227 |
| EquiLLM (Inv-Bridge + 1o) | 0.1940 | 0.653 | +0.5356 | 7.77e-07 | +0.9256 | +0.3095 |
| YLM-Universe FM (Ours) | 0.1754 | 0.875 | +0.6292 | 8.05e-07 | +0.9915 | +0.4963 |

### 4.2 Left vs. right handedness (`chiral_benchmark.json`)

| Model | Left/Right Acc (Test) ↑ | New Wording Acc ↑ | Unseen Molecule Acc ↑ | Mirror Flip Error ↓ | Flat Molecule Output ↓ | 3D Arrow Cos ↑ | 3x3 Matrix Cos ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GraphMVP / 3D-InfoMax | 52.1% | 47.9% | 50.0% | 6.92e-01 | 1.10e+00 | +0.0743 | +0.7201 |
| Uni-Mol (3D Dist-Transformer) | 52.1% | 52.1% | 48.4% | 7.62e-01 | 7.99e-01 | +0.0488 | +0.7218 |
| EquiLLM (Inv-Bridge + 1o) | 54.2% | 52.1% | 48.4% | 6.66e-01 | 7.35e-01 | +0.6574 | +0.9715 |
| YLM-Universe FM (Ours) | 100.0% | 100.0% | 100.0% | 4.27e-07 | 3.92e-08 | +0.7275 | +0.9982 |

What this run shows: the distance-only (`Uni-Mol`) and 2D graph (`GraphMVP`) models stay near 50% on mirror twins and fail the rotation check. `YLM-Universe` tells mirror twins apart in this tiny set and keeps 3D arrows correct under rotation, but it does not have the lowest energy MAE here.

---

## 5. Choosing Setup by GPU Size

- **Tier 1: Single 16–24 GB GPU (Colab T4 / L4, RTX 3090 / 4090)**
  - Freeze `MACE-OFF23-small` or `MACE-MP-0-small` (`--freeze_mace`) + train `YLM` (`option="C"`, `c_tp_mode="channelwise"`, `n_cross_layers=2`) + frozen `MatSciBERT`.
  - Start with a 200k–500k `SPICE-v2` subset.
- **Tier 2: Single 40–80 GB GPU (A100 / H100)**
  - Use `MACE-OFF23-medium` or `MACE-MP-0-medium`. Freeze MACE for the first 10 epochs, then unfreeze with a small learning rate (`1e-5`).
  - Fits full `SPICE-v2` (2M conformations) + `MPtrj` (1.5M crystal frames) + `Transition1x` reaction paths.
- **Tier 3: 4–8x A100 / H100**
  - Use `MACE-MP-0-large` across `OMol25` (83 elements) + `OMat24` + `Transition1x` + electrolyte datasets.

---

## 6. Datasets for Directions vs. Exact Sizes

| Dataset | What is inside | Labels | Role in training |
| :--- | :--- | :--- | :--- |
| **`SPICE-v2`** | 2M drug-like molecules, peptides, and `Li+/Na+/Mg2+/Cl-` water shells | `wB97M-D3BJ/def2-TZVPPD` energy, forces, dipole, quadrupole | Main organic & ion-shell source for both **direction and size** |
| **`OMol25`** | ~100M conformations across **83 elements** | `wB97M-V/def2-TZVPD` energy, forces, dipole | Broad element coverage (start with a 1M sample on 1 GPU) |
| **`MPtrj` + MP Dielectric** | 1.5M crystal frames (**89 elements**) + dielectric crystals | `PBE/PBE+U` forces, stress tensor, dielectric tensor | 3D repeating crystals and solid battery materials |
| **`Transition1x`** | 10M reaction path frames | `wB97x/6-31G(d)` barrier energies and forces | Sharpens **force directions** during bond breaking and forming |
| **`QM7-X` / `Wilkins QM7b`** | 4.2M conformations + CCSD polarizability matrices | Dipoles, forces, 3x3 polarizability matrices | High-accuracy 3x3 matrix supervision |

### Three Simple Training Rules When Mixing Datasets
1. **Split direction loss from length loss**: train cosine angle (`1 - cos`) separately from vector length (`| ||pred|| - ||true|| |`).
2. **Tag each quantum method (`[LOT=wb97m]`, `[LOT=pbe_u]`, `[LOT=xtb]`)**: fast `xtb` labels tend to get directions closer than lengths, so `xtb` rows train direction only (`length weight = 0`).
3. **Supervise which atom the text picks**: add a small cross-entropy loss (`--attn_loss_weight 0.2`) so questions about an `-OH` hydrogen focus on that `H` atom rather than sliding onto the neighbor `O` atom.

---

## 7. Step-by-Step Commands When GPU Arrives

### Step 1: Install and Run Sanity Checks (~10 min)
```bash
git clone https://github.com/yaniguan/YLM.git && cd YLM
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[data,crystals,mace,md,test]"

# 1) Run unit tests
pytest -q

# 2) Test real MACE-MP-0 on GPU with 3D rotation + mirror verification
ylm ask examples/li_ec_shell.xyz "What is the force on the lithium ion?" \
  --mace mace-mp-0-small --device cuda --verify-o3 --html gpu_mace_check.html
```

### Step 2: Prepare Data and Train `YLMUniverseModel` vs. `Uni-Mol`
```bash
python scripts/31_prepare_universe_data.py --inputs data/spice_v2.extxyz --lot wb97m --domain spice --out data/universe_spice

python scripts/30_train_universe_fm.py --name ylm_mace_mp0 --data_dir data/universe_spice \
  --backbone mace-mp-0-medium --freeze_mace --unfreeze_mace_epoch 10 \
  --text_encoder matscibert --epochs 30 --batch_size 64 --eval_rotations 32 --device cuda

python scripts/32_benchmark_vs_unimol.py --data_dir data/universe_spice \
  --ylm_backbone mace-mp-0-medium --text_encoder matscibert --epochs 30 --batch_size 64 --eval_rotations 32 --device cuda
```

### Step 3: Run Reactive MD with Live Event Plots
Run 20–50 ps on GPU with `OpenMMReactionReporter` or `ylm run-md`:
```bash
ylm run-md examples/proton_transfer_traj.extxyz --mace mace-off23-small --device cuda \
  --steps 20000 --dt-fs 0.5 --temp 300 \
  --out-traj runs/h5o2_mace.extxyz --plot runs/h5o2_timeline.svg --html runs/h5o2_report.html
```

### Step 4: Export Dataset Release for Hugging Face
```bash
ylm export-hf \
  --labels data/qm9_xtb.pkl \
  --v0 data/tensorqa_v0.json \
  --comp data/tensorqa_comp.json \
  --quad data/qm9_quad.pkl \
  --out dist/hf_tensorqa \
  --anonymized
```

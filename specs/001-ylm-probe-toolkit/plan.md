# Implementation Plan: YLM Directional Chemistry Probe & Toolkit

**Branch**: `001-ylm-probe-toolkit` | **Date**: 2026-11-10 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-ylm-probe-toolkit/spec.md`

## Summary

Build an easy-to-use 3D chemistry toolkit (`src/toolkit/` + `ylm` CLI) inside `yaniguan/YLM` that connects 3D symmetry features (`0e`, `1o`, `1e`, `2e`) with plain-English questions:
1. **Core 3D Motion Channel**: Optional `1o` step-to-step atom movement input (`n_motion_vectors` in `Encoder`, `use_motion_1o` in `YLMConfig`), working alongside `<vec>` 3D arrow readout (`vec_readout=True`) and periodic crystal boxes (`cell`, `pbc`).
2. **`YLMProbe` Runtime**: Runs packaged `checkpoints/v0/*.pt` (`YLM`) and fine-tuned `YLMCausalLM` models, returning 3D arrows in real units, word-to-atom attention weights, spherical channel norms, and live 3D rotation + mirror checks.
3. **Three Domain Packs**: `electrolyte` (ion shell counts, SSIP/CIP/AGG pairing, and escape direction), `md` (multi-frame `.extxyz` bond/proton event scanner + `1o` motion work), and `drug` (H-bond spots + clash-free 3D growth arrows).
4. **3-Level Post-Training Pipeline**: Built-in analytical force/dipole generator, optional xTB/ASE calculator, and `.extxyz` force reader for `IrrepsAdapter` + `<vec>` SFT and DPO training.
5. **Interactive 3D Reports & Web Studio**: Standalone HTML reports (`3Dmol.js` + built-in HTML5 `<canvas>` 3D viewer) and a local Web Studio (`ylm serve`) built with Python's standard library.

## Technical Context

**Language/Version**: Python 3.10+ (CI target: Python 3.11)

**Primary Dependencies**: `torch>=2.1`, `e3nn>=0.4.4`, `numpy>=1.24`, `scipy>=1.10`, `transformers>=4.40` (optional: `rdkit`, `ase`, `pymatgen`, `tblite`)

**Storage**: Local files (`.pt` checkpoints, `.xyz`/`.extxyz`/`.sdf`/`POSCAR`/`.cif` structures, `.json` configs/reports, standalone `.html` reports)

**Testing**: `pytest` (`python -m pytest -q`, runs offline on CPU)

**Target Platform**: Linux, macOS, Windows (CPU, CUDA, MPS)

**Project Type**: Python library + CLI + local Web Studio

**Performance Goals**: Single-molecule CPU inference + 3D symmetry check + HTML export under `2 s`; 20-frame MD trajectory scan under `5 s` on CPU.

**Constraints**: 100% backward compatibility with `checkpoints/v0/*.pt`; zero extra web server dependencies for `ylm serve`; works offline in CI.

**Scale/Scope**: Molecules, ion shells, protein pockets, and crystal cells up to ~256 atoms per frame; MD scans over hundreds of frames.

## Core Design Checks

- **1. Exact 3D Symmetry (`O(3)`)**: PASS — `extra_vectors` (`1o`), crystal `cell`, `<vec>` `1x1o` arrows, and analytical forces/dipoles all rotate cleanly as $\mathbf{v}(X R^\top + \mathbf{t}) = \mathbf{v}(X) R^\top$, while text scores and attention weights stay unchanged.
- **2. Backward Checkpoint Compatibility**: PASS — `n_motion_vectors=0` and `use_motion_1o=False` by default; `scripts/check_checkpoint_parity.py` passes with zero drift.
- **3. Library + CLI + Visual Output**: PASS — `src/toolkit/` Python API + `ylm` CLI + standalone HTML + `ylm serve`.
- **4. Works Offline**: PASS — Uses a deterministic token fallback when Hugging Face is offline, built-in force/dipole math when `tblite`/`rdkit` are not installed, and a `<canvas>` 3D viewer when offline.
- **5. Clear Physical Units**: PASS — Converts cleanly between `bohr`, `Å`, `Debye`, and `eV/Å`.

## Project Structure

### Documentation

```text
.specify/
├── feature.json
└── memory/
    └── constitution.md

specs/001-ylm-probe-toolkit/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── encoder.py                 # + optional n_motion_vectors (1o extra_vectors) & PBC cell
├── irreps_to_text.py          # Option A/B/C + channelwise TP
├── model.py                   # YLM v0 architecture
├── ylm_lm.py                  # YLMCausalLM + vec_readout + use_motion_1o + delta_pos collation
├── tasks/                     # Benchmark & SFT/DPO utilities
└── toolkit/                   # YLM Probe & Domain Toolkit
    ├── __init__.py            # Public Python API exports
    ├── ingest.py              # SMILES, formulation, .xyz/.extxyz, .sdf, POSCAR/.cif loader
    ├── probe.py               # YLMProbe (v0 + CausalLM), attention & live O(3) checker
    ├── domains.py             # electrolyte, md, and drug domain helpers
    ├── posttrain.py           # 3-level physics labeler + Adapter/<vec> SFT & DPO
    ├── viz.py                 # Standalone 3Dmol.js + offline Canvas 3D HTML viewer
    ├── server.py              # Standard-library HTTP server for Web Studio (ylm serve)
    └── cli.py                 # `ylm` CLI entry point

examples/
├── water.xyz                  # Sample polar molecule
├── li_ec_shell.xyz            # Sample Li+–ethylene carbonate solvation shell
├── proton_transfer_traj.extxyz# Sample 4-frame proton-transfer MD trajectory
└── indole_pocket.xyz          # Sample drug fragment + protein pocket atoms

tests/
├── test_toolkit_ingest_probe.py
├── test_toolkit_domains_motion.py
└── test_toolkit_posttrain_viz_cli.py
```

**Structure Decision**: Keep one clean repository extending `src/` with `src/toolkit/` and exposing `[project.scripts] ylm = "src.toolkit.cli:main"` in `pyproject.toml`, leaving existing experiment scripts (`scripts/01_...`–`23_...`) untouched.

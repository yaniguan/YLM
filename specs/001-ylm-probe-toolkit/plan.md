# Implementation Plan: YLM Directional Chemistry Probe & Domain Toolkit

**Branch**: `001-ylm-probe-toolkit` | **Date**: 2026-11-10 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-ylm-probe-toolkit/spec.md`

## Summary

Build an out-of-the-box directional chemistry toolkit (`src/toolkit/` + `ylm` CLI) inside `yaniguan/YLM` that fuses $O(3)$-equivariant `irreps` (`0e + 1o + 1e + 2e`) with language tokens across:
1. **Core Equivariant Extension**: Opt-in `1o` inter-frame displacement/velocity input channel (`n_motion_vectors` in `Encoder`, `use_motion_1o` in `YLMConfig`), building on channel-wise Option C (`c_tp_mode`), `<vec>` readout (`vec_readout=True`), and PBC crystal graphs (`cell`, `pbc`).
2. **Dual-Engine `YLMProbe` Runtime**: Instant inference over packaged `checkpoints/v0/*.pt` (`YLM`) and fine-tuned `YLMCausalLM` (`xattnA/B/C` + `<vec>`), returning physical 3D vectors, token-to-atom routing matrices, spherical-harmonic `l`-channel breakdowns, and live $O(3)$ rotation+reflection verification.
3. **Three Domain Packs**: `electrolyte` (solvation-shell coordination, SSIP/CIP/AGG classification, ion escape direction), `md` (multi-frame `.extxyz` bond/proton event scanner + `1o` $\Delta \mathbf{r}$ features), and `drug` (pharmacophore anchors + pocket-clash-filtered 3D substituent exit vectors).
4. **3-Tier Post-Training Pipeline**: Offline analytical/MMFF physics generator, optional xTB/ASE calculator hook, and `.extxyz` force importer driving `IrrepsAdapter` + `<vec>` SFT and counterfactual DPO.
5. **Interactive Visualization & Studio**: Zero-dependency `3Dmol.js` + HTML5 `<canvas>` 3D standalone HTML reports and stdlib `http.server` Web Studio (`ylm serve`).

## Technical Context

**Language/Version**: Python 3.10+ (CI target: Python 3.11)

**Primary Dependencies**: `torch>=2.1`, `e3nn>=0.4.4`, `numpy>=1.24`, `scipy>=1.10`, `transformers>=4.40` (optional: `rdkit`, `ase`, `pymatgen`, `tblite`)

**Storage**: Local files (`.pt` checkpoints, `.xyz`/`.extxyz`/`.sdf`/`POSCAR`/`.cif` structures, `.json` configs/reports, standalone `.html` reports)

**Testing**: `pytest` (`python -m pytest -q`, offline CPU-compatible)

**Target Platform**: Linux, macOS, Windows (CPU, CUDA, MPS)

**Project Type**: Library + CLI + Local Interactive Web Studio (Monorepo)

**Performance Goals**: Single-molecule/cluster CPU inference + $O(3)$ verification + HTML export $< 2\text{ s}$; 20-frame MD trajectory scan $< 5\text{ s}$ on CPU.

**Constraints**: Strict backward compatibility with `checkpoints/v0/*.pt`; zero required web-framework dependencies for `ylm serve`; full offline operation in air-gapped CI.

**Scale/Scope**: Molecules, solvation shells, binding pockets, and crystal unit cells up to ~256 atoms per frame; MD trajectory scans over hundreds of frames.

## Constitution Check

*GATE: Passed before Phase 0 research and re-verified after Phase 1 design.*

- **Principle I (Exact $O(3)$ Equivariance)**: PASS — `extra_vectors` (`1o`), PBC `cell`, `<vec>` `1x1o` readout, and Tier-1 analytical forces/dipoles all transform strictly under $D^{1o}(R) = R$, while text logits and attention weights use only $O(3)$ invariants.
- **Principle II (Backward Checkpoint Parity)**: PASS — `n_motion_vectors=0` and `use_motion_1o=False` by default; `scripts/check_checkpoint_parity.py` verified unchanged.
- **Principle III (Library + CLI + Visual Observability)**: PASS — `src/toolkit/` SDK + `ylm` CLI + standalone HTML + `ylm serve`.
- **Principle IV (Offline-Safe Reproducibility)**: PASS — Deterministic invariant token hash fallback when HF hub is offline; pure-NumPy/PyTorch analytical force/dipole fallback when `tblite`/`rdkit` unavailable; HTML5 `<canvas>` 3D fallback when CDN is offline.
- **Principle V (Physical Unit Transparency)**: PASS — Explicit conversions between `bohr`, `Å`, `Debye`, and `eV/Å`.

## Project Structure

### Documentation (this feature)

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
├── tasks/                     # Existing benchmark & SFT/DPO utilities
└── toolkit/                   # Out-of-the-box YLM Probe & Domain Application Suite
    ├── __init__.py            # Public Python SDK exports
    ├── ingest.py              # SMILES, formulation, .xyz/.extxyz, .sdf, POSCAR/.cif ingestor
    ├── probe.py               # Dual-engine YLMProbe (v0 + CausalLM), routing & live O(3) verifier
    ├── domains.py             # electrolyte, md, and drug specialized domain packs
    ├── posttrain.py           # 3-tier physics labeler + Adapter/<vec> SFT & counterfactual DPO
    ├── viz.py                 # Standalone 3Dmol.js + offline Canvas 3D arrow/attention HTML exporter
    ├── server.py              # Zero-dep ThreadingHTTPServer Web Studio (ylm serve)
    └── cli.py                 # Unified `ylm` CLI entry point

examples/
├── water.xyz                  # Demo polar molecule
├── li_ec_shell.xyz            # Demo Li+–ethylene carbonate solvation shell
├── proton_transfer_traj.extxyz# Demo 4-frame proton-transfer MD snippet
└── indole_pocket.xyz          # Demo drug fragment + pocket atoms

tests/
├── test_toolkit_ingest_probe.py
├── test_toolkit_domains_motion.py
└── test_toolkit_posttrain_viz_cli.py
```

**Structure Decision**: Single monorepo extending `src/` with `src/toolkit/` and exposing `[project.scripts] ylm = "src.toolkit.cli:main"` in `pyproject.toml`, keeping existing research scripts (`scripts/01_...`–`23_...`) untouched.

---
description: "Task list for YLM Directional Chemistry Probe & Domain Toolkit"
---

# Tasks: YLM Directional Chemistry Probe & Domain Toolkit

**Input**: Design documents from `/specs/001-ylm-probe-toolkit/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `quickstart.md`, `literature/*.md`

## Phase 1: Setup & Core Equivariant Motion Extension

- [x] T001 Extend `src/encoder.py` (`Encoder`) with optional `n_motion_vectors: int = 0` and `extra_vectors` (`1o`) input channel while preserving `strict=True` v0 checkpoint parity
- [x] T002 Extend `src/ylm_lm.py` (`YLMConfig`, `collate_geometry`, `IrrepsTokenizer`, `YLMCausalLM`) with `use_motion_1o: bool = False`, `delta_pos` collation, and optional adapter routing inspection cache
- [x] T003 Update `pyproject.toml` to register `[project.scripts] ylm = "src.toolkit.cli:main"` and include `src.toolkit` in `packages`

---

## Phase 2: Foundational Ingestion & Dual-Engine Probe (US1 - P1 MVP)

- [x] T004 [P] [US1] Implement `src/toolkit/ingest.py` (`StructureFrame`, `load_structure`, `load_trajectory`, `build_solvation_cluster`, `.xyz`/`.extxyz`/`.sdf`/`POSCAR`/`.cif`/SMILES parsers, minimum-image `delta_pos_bohr`)
- [x] T005 [US1] Implement `src/toolkit/probe.py` (`YLMProbe` for `v0` + `YLMCausalLM`, offline-safe text encoder, heavy-element valence-group fallback, `l`-channel norm breakdown, `verify_o3_equivariance`)
- [x] T006 [P] [US1] Create sample structures in `examples/` (`water.xyz`, `li_ec_shell.xyz`, `proton_transfer_traj.extxyz`, `indole_pocket.xyz`)
- [x] T007 [US1] Add unit & equivariance tests in `tests/test_toolkit_ingest_probe.py`

---

## Phase 3: 120-Paper Literature Survey & Specialized Domain Packs (US2 - P1)

- [x] T008a [P] [US2] Survey and archive 40 verified papers per application domain (`literature/01_drug_sbdd_3dmollm.md`, `02_electrolyte_battery_ai.md`, `03_equivariant_md_dynamics_llm.md`)
- [x] T008 [US2] Implement literature-aligned `src/toolkit/domains.py` covering `analyze_electrolyte` (discrete & smooth CN, hard & soft SSIP/CIP/AGG, ligand radial $\cos\theta$, `1o` escape vector, centrosymmetric `2e` channel axis), `scan_md_trajectory` (`SEGNN` $\mathbf{1o}\otimes\mathbf{1o}\to \mathbf{0e}\oplus\mathbf{1e}\oplus\mathbf{2e}$ work/torque/virial, Grotthuss $\delta(t)$ + smooth CN + rattling filter, saliency keyframing), and `analyze_drug` (`2e` ring plane tensor, HBD/HBA cones + pocket H-bond lock, open-valence heavy-atom fallback, ray-marching exit vectors, multi-fragment Linker bridges)
- [x] T009 [US2] Add domain & `1o`/`1e`/`2e` equivariance tests in `tests/test_toolkit_domains_motion.py`

---

## Phase 4: 3-Tier Physics Labeler & `IrrepsAdapter` + `<vec>` Post-Training (US3 - P2)

- [x] T010 [US3] Implement `src/toolkit/posttrain.py` (`PhysicsLabeler` supporting analytical/MMFF, optional xTB/ASE, and `.extxyz` force import; `build_directional_sft_examples`; `build_directional_dpo_pairs`; `run_posttrain` with `--freeze-lm` and reloadable checkpoint export)

---

## Phase 5: Standalone 3D HTML Visualizer, Local Web Studio, HF Exporter & Unified CLI (US4 - P2)

- [x] T011 [P] [US4] Implement `src/toolkit/viz.py` (`render_html_report`, `render_trajectory_html_report` with `3Dmol.js` + built-in rotatable HTML5 `<canvas>` 3D fallback, 3D vector arrows, clickable token-atom attention heatmap, `0e/1o/1e/2e` bars, and O(3) badge)
- [x] T012 [P] [US4] Implement `src/toolkit/server.py` (`ThreadingHTTPServer` Web Studio serving `GET /`, `GET /api/health`, `POST /api/ask`) and `src/toolkit/hf_export.py` (`export_tensorqa_hf` + bundled $K=32$ $O(3)$ evaluation script)
- [x] T013 [US4] Implement `src/toolkit/cli.py` (`ylm ask`, `scan-traj`, `solvation`, `drug`, `posttrain`, `verify-o3`, `serve`, `export-hf`) and `src/toolkit/__init__.py`
- [x] T014 [US3] [US4] Add end-to-end post-training, HTML visualization, HTTP server, CLI, and HF export tests in `tests/test_toolkit_posttrain_viz_cli.py`

---

## Phase 6: Polish, Documentation & Verification

- [x] T015 Update `README.md` with the `ylm` toolkit quickstart, domain commands, and Spec-Kit specification pointers
- [x] T016 Run full `pytest -q` suite and `scripts/check_checkpoint_parity.py`, mark all tasks `[x]`, push branch `001-ylm-probe-toolkit`, and open GitHub PR

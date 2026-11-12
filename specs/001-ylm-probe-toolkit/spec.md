# Feature Specification: YLM Directional Chemistry Probe & Domain Toolkit

**Feature Branch**: `001-ylm-probe-toolkit`

**Created**: 2026-11-10

**Status**: Approved

**Input**: User description: "Combine 3D directional atom features with language model tokens so the model can read and output 3D directions across drug molecules, battery electrolytes, and reactive MD trajectories, with an easy CLI, Python SDK, 3D HTML viewer, and local browser app."

## Clarifications

### Session 2026-11-10

- Q: How should the toolkit be organized in Git? → A: Keep everything in the `yaniguan/YLM` repository using feature branches and pull requests.
- Q: Which model runtime should power inference? → A: Support two engines: fast `checkpoints/v0` vector probes (ready right away with no training) and `YLMCausalLM` + `IrrepsAdapter` + `<vec>` generation (for local fine-tuning and Hugging Face models).
- Q: How should drug, electrolyte, and MD workflows share geometry features? → A: Add an optional frame-to-frame motion vector channel (`1o`) to the 3D encoder and build `drug`, `electrolyte`, and `md` topic helpers on top.
- Q: How should post-training generate force and dipole labels when quantum packages are missing? → A: Provide three label options: built-in analytical/MMFF forces (works offline anywhere), optional xTB/ASE quantum labels, and importing pre-labeled `.extxyz` trajectories.
- Q: What user interfaces should be included? → A: `ylm` CLI commands, Python functions, single-file 3D HTML reports (`3Dmol.js` + offline `<canvas>` fallback), and a local browser app (`ylm serve`).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Zero-Training 3D Directional Probe & O(3) Verification (Priority: P1)

A researcher has a molecular structure (`.xyz`, `.sdf`, `.poscar`, or SMILES) and wants to ask a natural-language directional question (e.g., *"Which way does the dipole point?"* or *"What is the force vector on the hydroxyl hydrogen?"*) immediately after `pip install -e .`, getting back unit-labeled 3D covariant vectors, per-token atom-attention routing weights, spherical-harmonic `l`-channel contributions (`0e/1o/1e/2e`), a live $O(3)$ rotation+reflection self-check badge, and a standalone interactive 3D HTML report.

**Why this priority**: Works right away in under 10 seconds using packaged `checkpoints/v0/*.pt` or saved `YLMCausalLM` checkpoints without requiring training or external software.

**Independent Test**: Run `ylm ask examples/water.xyz "What is the dipole vector?" --verify-o3 --html report.html` offline on CPU; verify non-zero 3D vector output, $O(3)$ equivariance error $< 10^{-5}$, and valid self-contained HTML generation.

**Acceptance Scenarios**:

1. **Given** a `.xyz` file or SMILES string and a packaged `v0` checkpoint, **When** the user calls `YLMProbe.default().ask(...)` or `ylm ask`, **Then** the system returns physical 3D vectors (`Debye` / `eV/Å`), top attended atoms, `l`-channel norms, and optional $O(3)$ verification metrics.
2. **Given** an input structure containing elements with $Z \ge 10$ evaluated on a legacy `n_elements=10` `v0` checkpoint, **When** `probe.ask(...)` runs, **Then** heavy elements are mapped by valence group without crashing and an explicit warning is recorded in `ProbeResult.warnings`.

---

### User Story 2 - Electrolyte, MD Trajectory (`1o` Motion), and Drug Domain Packs (Priority: P1)

Domain scientists working on (a) electrolyte solvation shells / periodic battery electrodes, (b) multi-frame MD trajectories, or (c) ligand-pocket drug structures use specialized domain workflows that combine `1o` motion channels, PBC crystal cells, and role-aware geometry with YLM directional queries.

**Why this priority**: Demonstrates why directional `irreps`-to-text fusion solves real chemistry problems where 2D SMILES or scalar-only GNNs fail.

**Independent Test**: Run `ylm solvation`, `ylm scan-traj`, and `drug` pharmacophore/exit-vector analysis on offline synthetic/demo inputs and verify $O(3)$ covariance of all domain-recommended 3D vectors and inter-frame `1o` displacement equivariance.

**Acceptance Scenarios**:

1. **Given** an electrolyte solvation cluster or periodic electrode cell (`cell`, `pbc`), **When** running the `electrolyte` pack, **Then** the system identifies coordinating ligands around the working ion (SSIP/CIP/AGG motif), computes cage pull / desolvation escape vectors, and runs YLM directional probing.
2. **Given** a multi-frame `.extxyz` MD trajectory, **When** running `ylm scan-traj` or `Encoder(..., n_motion_vectors=1)`, **Then** inter-frame displacements $\Delta \mathbf{r}_i(t)$ transform as `1o` odd vectors under $O(3)$, bond-stretch/proton-transfer frames are detected, and directional force-displacement alignment $\cos\theta$ is reported.
3. **Given** a drug ligand (with optional binding-pocket atoms marked via `role`), **When** running the `drug` pack, **Then** the system identifies H-bond / aromatic / C–H anchors, filters pocket steric clashes, and outputs ranked 3D substituent exit vectors `<vec>` plus YLM polar directional vectors.

---

### User Story 3 - Lightweight `IrrepsAdapter` + `<vec>` Post-Training Pipeline (Priority: P2)

A practitioner wants to fine-tune `YLMCausalLM` (`xattnA/B/C` with `vec_readout=True` and optional `use_motion_1o=True`) on their own structures or trajectories so the LM emits domain-specific natural-language explanations interleaved with `<vec>` 3D directional predictions.

**Why this priority**: Enables turning any pretrained text backbone (local `tiny` GPT-2 or HuggingFace causal LM with `--freeze-lm`) into a directional chemistry assistant in minutes.

**Independent Test**: Run `ylm posttrain` on a small offline molecule/trajectory set using the built-in zero-dependency force/dipole generator, save a checkpoint directory, reload it via `YLMProbe.from_causal_lm(...)`, and verify loss decrease + exact $O(3)$ equivariance of decoded `<vec>` vectors.

**Acceptance Scenarios**:

1. **Given** unlabeled SMILES or `.xyz` structures on a machine without `tblite`, **When** `ylm posttrain --backend auto` runs, **Then** the Tier-1 analytical/MMFF physics backend generates covariant 3D forces and dipoles, builds SFT `<vec>` examples (and optional counterfactual DPO pairs), trains `YLMCausalLM`, and writes a reloadable run directory.
2. **Given** a pre-labeled `.extxyz` trajectory with `forces`, **When** `ylm posttrain --backend extxyz` runs, **Then** the Tier-3 importer constructs atom-resolved natural-language directional QA + `<vec>` supervision directly from the recorded forces.

---

### User Story 4 - Interactive 3D HTML Report & Local Web Studio (`ylm serve`) (Priority: P2)

Users inspect directional vectors and token-to-atom routing visually in a browser either by opening a generated standalone `.html` file or launching `ylm serve` to upload structures and type questions interactively.

**Why this priority**: Makes 3D vectors and `IrrepsToText` routing immediately tangible without PyMOL, VMD, or Node.js build steps.

**Independent Test**: Generate an HTML report via `res.to_html(...)` and hit `GET /`, `GET /api/health`, and `POST /api/ask` on `ylm serve` in an automated test; verify valid HTML/JSON payloads containing 3D coordinates, arrow endpoints, attention matrices, and $O(3)$ check badges.

**Acceptance Scenarios**:

1. **Given** a `ProbeResult`, **When** `to_html()` is called, **Then** it produces a single self-contained HTML file featuring `3Dmol.js` + an offline HTML5 `<canvas>` 3D rotatable fallback viewer, 3D arrow overlays, clickable token chips that highlight attended atoms, and `0e/1o/1e/2e` channel bars.
2. **Given** `ylm serve` running locally, **When** a client posts a structure and question to `/api/ask`, **Then** the server returns structured JSON with vectors, text answer, routing weights, and $O(3)$ self-check status.

### Edge Cases

- **Offline / air-gapped execution**: When HuggingFace `MiniLM`/`MatSciBERT` weights or `3Dmol.js` CDN are unreachable, `YLMProbe` falls back to a deterministic $O(3)$-invariant token hash encoder (recording a warning) and the HTML viewer falls back to its built-in interactive HTML5 `<canvas>` 3D renderer.
- **Heavy elements on legacy `v0` checkpoints**: Elements with $Z \ge 10$ are mapped to valence-group analogues in `1..9` with an explicit warning; `YLMCausalLM` (`n_elements=100`) handles $Z=1..99$ natively.
- **Single-frame vs. multi-frame inputs**: When a single static structure is fed to a `use_motion_1o=True` model, `delta_pos` defaults cleanly to zeros `(B, N, 3)`, preserving exact $O(3)$ equivariance.
- **Degenerate or 1-atom geometries**: Single-atom inputs produce empty neighbor edge sets without NaNs and return finite vectors.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `Encoder` and `IrrepsTokenizer` MUST support an opt-in `1o` equivariant per-atom vector input channel (`n_motion_vectors=0` default; `use_motion_1o=True` in `YLMConfig`) for inter-frame displacements $\Delta \mathbf{r}_i$ or velocities while keeping `strict=True` state-dict compatibility when disabled.
- **FR-002**: `src/toolkit/ingest.py` MUST ingest SMILES strings, multi-species electrolyte recipes, `.xyz` / multi-frame `.extxyz` (including `Lattice`, `forces`, `vel`), `.sdf`/`.mol`, and `POSCAR`/`.cif` crystals into normalized `StructureFrame` objects with Bohr and Ångström coordinates.
- **FR-003**: `src/toolkit/probe.py` MUST provide `YLMProbe` supporting both packaged `v0` checkpoints (`YLM`) and `YLMCausalLM` (`vec_readout=True`) run directories, returning `ProbeResult` with unit-converted 3D vectors, token-to-atom routing matrices, and per-`l` channel norm breakdowns.
- **FR-004**: `src/toolkit/probe.py` MUST provide live $O(3)$ equivariance verification (`verify_o3_equivariance`) testing random improper/proper orthogonal matrices $R \in O(3)$ and translations $\mathbf{t} \in \mathbb{R}^3$.
- **FR-005**: `src/toolkit/domains.py` MUST implement literature-aligned domain analyzers and directional query builders grounded in 120 surveyed papers (`specs/001-ylm-probe-toolkit/literature/`): `electrolyte` (discrete & smooth coordination numbers, hard & soft SSIP/CIP/AGG speciation, ligand dipole radial alignment $\cos\theta$, `1o` desolvation escape vector, and centrosymmetric `2e` quadrupole channel axis), `md` (`SEGNN` $\mathbf{1o}\otimes\mathbf{1o}\to \mathbf{0e}\oplus\mathbf{1e}\oplus\mathbf{2e}$ mechanical work $W_i$, axial torque, virial shear, signed Grotthuss $\delta(t)$ + smooth rational CN + committed vs. rattling classification, and event saliency keyframing), and `drug` (`2e` aromatic ring plane tensor, HBD + lone-pair HBA cones with pocket H-bond lock detection, open-valence heavy-atom fallback, multi-step vdW ray-marching exit vectors, and multi-fragment Linker/PROTAC bridge quartets).
- **FR-006**: `src/toolkit/posttrain.py` MUST implement a 3-tier directional label generator (`analytical/mmff`, optional `xtb/ase`, and `extxyz` importer) plus `build_directional_sft_examples`, `build_directional_dpo_pairs`, and `run_posttrain` with `--freeze-lm` support, saving standard `YLMCausalLM` run directories.
- **FR-007**: `src/toolkit/viz.py` MUST render standalone interactive HTML reports with 3D molecular/cell rendering (`3Dmol.js` + offline `<canvas>` 3D fallback), 3D directional arrows, interactive token-atom attention highlighting, `l`-channel bars, and $O(3)$ verification badges.
- **FR-008**: `src/toolkit/server.py` MUST provide a zero-extra-dependency HTTP server (`ylm serve`) serving an interactive browser studio (`GET /`) and JSON endpoints (`GET /api/health`, `POST /api/ask`).
- **FR-009**: `src/toolkit/cli.py` and `pyproject.toml` MUST expose the `ylm` console script with subcommands `ask`, `scan-traj`, `solvation`, `drug`, `posttrain`, `verify-o3`, `serve`, and `export-hf`.
- **FR-010**: `src/toolkit/hf_export.py` MUST export `TensorQA` (`v0_core`, optional `v0_compositional` T4/T5, and optional `v0_quadrupole` $l=2$) into self-contained Hugging Face Dataset JSONL splits with a complete Dataset Card (`--anonymized` option for double-blind review) and standalone $K=32$ $O(3)$ evaluation script (`evaluate_o3_protocol.py`).
- **FR-011**: `src/mace_backbone.py` (`MACEEncoder`) MUST adapt pretrained `MACE-MP-0` (89 elements) and `MACE-OFF23` foundation models into both `YLM` (`YLM.from_mace`) and `YLMCausalLM` (`YLMConfig(encoder="mace-...")`), preserving exact $O(3)$ equivariance across `node_feats` and conservative atomic force (`1x1o`) channels under molecular and periodic boundary conditions.
- **FR-012**: `src/toolkit/md_hooks.py` MUST provide `LiveReactionMonitor`, `OpenMMReactionReporter`, `LAMMPSReactionHook`, and `plot_reaction_timeline` (`--plot` on `ylm scan-traj`) for streaming detection and timestamped plotting of bond-forming, bond-breaking, and committed proton-transfer events during MD simulations.

### Key Entities

- **`StructureFrame`**: Unified 3D chemical snapshot holding `Z`, `pos_bohr`, `cell_bohr`, `pbc`, `role`, `amount`, `species`, optional `delta_pos_bohr` (`1o`), and optional reference `forces_ev_ang` / `dipole_debye`.
- **`VectorPrediction`**: Named 3D vector output with Cartesian components in atomic and physical units (`Debye`, `eV/Å`, or unit vector), anchor coordinates (atom index or centroid), and norm.
- **`ProbeResult`**: Complete response from `YLMProbe.ask` combining natural-language `answer`, `vectors`, `tokens`, `atom_attention` `(T, N)`, `channel_norms` (`0e, 0o, 1o, 1e, 2e`), `o3_check`, `domain_summary`, and `warnings`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Exact $O(3)$ equivariance holds to $< 10^{-7}$ in `float64` (`< 5 \times 10^{-5}` in `float32`) for `Encoder` with `n_motion_vectors=1` + PBC `cell`, `YLMProbe` (`v0` and `YLMCausalLM`), and Tier-1 physics vector labels.
- **SC-002**: All packaged `checkpoints/v0/*.pt` files continue to pass `scripts/check_checkpoint_parity.py` with zero missing/unexpected keys.
- **SC-003**: Users can run `ylm ask`, `ylm solvation`, `ylm scan-traj`, `ylm drug`, `ylm posttrain`, and `ylm verify-o3` end-to-end on CPU in under 15 seconds without network access.
- **SC-004**: Post-training `YLMCausalLM` (`vec_readout=True`) for 15 steps on a toy directional dataset measurably decreases combined text + `<vec>` MSE loss and produces reloadable checkpoints whose decoded `<vec>` vectors rotate covariantly under $O(3)$.

## Assumptions

- Users installing core dependencies (`pip install -e .`) have Python $\ge 3.10$, `torch`, `e3nn`, `numpy`, `scipy`, and `transformers`.
- Optional packages (`rdkit`, `ase`, `pymatgen`, `tblite`) enrich conformer embedding and quantum labels when installed, while offline parser and analytical force-field fallbacks guarantee every CLI command works right away without them.

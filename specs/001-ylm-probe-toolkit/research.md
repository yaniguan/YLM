# Phase 0 Research & Literature Notes: YLM Directional Chemistry Toolkit

Detailed paper tables for each part of the toolkit live in `literature/`:
- **Drug design, 3D growth arrows & molecule LLMs (40 papers)**: [`literature/01_drug_sbdd_3dmollm.md`](./literature/01_drug_sbdd_3dmollm.md)
- **Battery electrolytes, crystal ion channels & battery LLMs (40 papers)**: [`literature/02_electrolyte_battery_ai.md`](./literature/02_electrolyte_battery_ai.md)
- **Molecular dynamics, proton hops & MD agents (40 papers)**: [`literature/03_equivariant_md_dynamics_llm.md`](./literature/03_equivariant_md_dynamics_llm.md)
- **2D chemical drawings (`Ouroboros-OCSR`), 3D handedness & ML force fields (38 papers)**: [`literature/04_ocsr_2d_to_3d_stereochem_mlip.md`](./literature/04_ocsr_2d_to_3d_stereochem_mlip.md)

## Decision 1: Step-to-Step Motion Arrows (`1o`) and Work Breakdown in MD
- **Background**: `EGNN` (2021), `SEGNN` (2022), Calegari Andrade et al. (2020), `mlcolvar` (2023), `ZundEig` (2023), Gomez, Thompson & Laage (*Nat. Chem.* 2024), `EquiLLM` (2025).
- **What we built**:
  1. Added optional `n_motion_vectors: int = 0` to `Encoder.__init__` and `extra_vectors=None` to `Encoder.forward` (`use_motion_1o: bool = False` in `YLMConfig`). When turned on, frame-to-frame atom steps $\Delta\mathbf{r}_i$ (wrapped cleanly across periodic box edges) enter Layer 0 as `1o` 3D vectors. When turned off (default), older `v0` checkpoints load unchanged.
  2. In `analyze_md_frame_motion`, combined force $\mathbf{F}_i$ and step $\Delta\mathbf{r}_i$ into mechanical work $W_i = \mathbf{F}_i \cdot \Delta\mathbf{r}_i$ (`0e`), rotational torque $\mathbf{F}_i \times \Delta\mathbf{r}_i$ (`1e`), and local shear (`2e`). In `detect_md_frame_events`, tracked the proton-sharing number $\delta(t) = d(\text{D-H}^*) - d(\text{A-H}^*)$ to separate real proton hops (`committed_grotthuss_transfer`) from back-and-forth heat vibrations (`frustrated_rattling`).

## Decision 2: Electrolyte Escape Arrows (`1o`) and Symmetric Crystal Channels (`2e`)
- **Background**: `SolvationAnalysis` (2023), Holoubek & Liu (*Nat. Energy* 2021), `CALiSol-23` (2024), `BAMBOO` (2024), `MACE-MP-0` (2025), `LiTraj` (2025), `DiffMix` (2024), `ChatBattery` (2025).
- **What we built**: Combined smooth coordination counts, SSIP / CIP / AGG ion-pairing scores (`0e`), solvent-dipole alignment $\cos\theta_k$ (`0e`), a `1o` 3D escape arrow pointing out of a broken solvent cage, and a `2e` 3x3 symmetric cage matrix whose least-crowded axis $\pm\hat{\mathbf{u}}_{\text{open}}$ still finds open ion tunnels inside symmetric crystals where `1o` arrows cancel to zero.

## Decision 3: Drug Ring Planes (`2e`), H-Bonds, and Safe 3D Growth Arrows
- **Background**: `ShEPhERD` (2025), `PharmacoNet` (2024), `SurfDock` (2025), `PocketXMol` (2026), `3DLinker` (2022), `DiffLinker` (2024), `3D-MoLM` (2024), `BindGPT` (2025).
- **What we built**: Represented flat 5- and 6-membered rings with a sign-free `2e` 3x3 plane matrix, found H-bond donors (`N-H`/`O-H`) and lone-pair acceptors (`1o`) while flagging H-bonds already locked to the protein, checked van der Waals clearance from `1.5 Å` to `4.2 Å` along candidate C-H rays to pick safe growth directions, and measured linker bridge gaps when two fragments sit in one pocket.

## Decision 4: Offline-Safe Dual Engine, Physics Labeler, and Benchmark Export
- **What we built**: Supported both packaged `checkpoints/v0/*.pt` (`YLM`) and fine-tuned `YLMCausalLM` (`vec_readout=True`), a 3-level physics labeler (`.extxyz` forces -> `xtb` -> built-in analytical force & dipole math), single-file HTML 3D viewers (`3Dmol.js` + built-in `<canvas>` fallback), a standard-library local web server (`ylm serve`), and a Hugging Face `TensorQA` dataset exporter (`ylm export-hf`).

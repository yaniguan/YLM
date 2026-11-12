# YLM Constitution

## Core Principles

### I. Exact O(3) Equivariance & Invariance (NON-NEGOTIABLE)
Every geometric pathway MUST respect 3D Euclidean + parity symmetries ($O(3)$):
- Natural-language token probabilities, scalar predictions, and cross-attention weights MUST remain invariant under global rotations, improper reflections ($\det R = -1$), and translations.
- Predicted 3D directional vectors (`1o`, e.g., dipole moments, per-atom forces, ion escape directions, pharmacophore exit vectors) MUST transform covariantly as $\mathbf{v}(X R^\top + \mathbf{t}) = \mathbf{v}(X) R^\top$.

### II. Backward Checkpoint Parity
All changes to core model modules (`src/encoder.py`, `src/irreps_to_text.py`, `src/model.py`, `src/ylm_lm.py`) MUST be opt-in via default-disabled constructor arguments so packaged checkpoints (`checkpoints/v0/*.pt` and Oct-30 runs) load with `strict=True` and zero numerical drift.

### III. Library-First + CLI + Zero-Dependency Visual Observability
Every user-facing capability in `src/toolkit/` MUST be exposed first as a clean, typed Python API, wrapped by the unified `ylm` CLI (`pyproject.toml` entry point), and inspectable via standalone single-file `3Dmol.js` + HTML5 Canvas 3D vector/attention reports and a zero-extra-dependency stdlib Web Studio (`ylm serve`).

### IV. Offline-Safe Reproducibility & Test-First Verification
All unit and integration tests under `tests/` MUST execute deterministically on CPU without network access, paid APIs, or optional quantum-chemistry binaries (`tblite`, `pymatgen`, `mp-api`). External calculators and HuggingFace hub downloads MUST degrade gracefully to deterministic offline fallbacks when unavailable.

### V. Physical Unit Transparency
Internal coordinates use atomic units (`bohr`), while user-facing SDKs, CLI outputs, and HTML reports MUST explicitly label and convert physical units (`Å`, `Debye`, `eV/Å`, `Ha/bohr`).

## Architectural Constraints

- **Monorepo Layout**: Core equivariant modules remain in `src/`, benchmark tasks in `src/tasks/`, and the user-facing application toolkit lives in `src/toolkit/`.
- **Core Dependencies**: Only `torch`, `e3nn`, `numpy`, `scipy`, and `transformers` are required at runtime; `rdkit`, `ase`, `pymatgen`, and `tblite` are optional extras.

## Development Workflow & Quality Gates

- All new features follow Spec-Kit specifications under `specs/<NNN-feature>/` (`spec.md`, `plan.md`, `tasks.md`).
- Pull requests MUST pass `python -m pytest -q` on Python 3.11 CPU CI and `scripts/check_checkpoint_parity.py`.

## Governance

This Constitution governs architectural and implementation decisions across `yaniguan/YLM`. Any deviation from exact $O(3)$ symmetry (e.g., frame-canonicalization fallbacks on degenerate point clouds or element remapping for legacy `n_elements=10` checkpoints) MUST be surfaced explicitly in result metadata/warnings.

**Version**: 1.0.0 | **Ratified**: 2026-11-10 | **Last Amended**: 2026-11-10

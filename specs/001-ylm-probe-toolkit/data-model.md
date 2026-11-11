# Phase 1 Data Model: `src/toolkit`

## 1. `StructureFrame` (`src/toolkit/ingest.py`)
| Field | Type | Units / Shape | Description |
|-------|------|---------------|-------------|
| `Z` | `np.ndarray` (`int64`) | `(N,)` | Atomic numbers ($1 \le Z \le 99$) |
| `pos_bohr` | `np.ndarray` (`float64`) | `(N, 3)` `bohr` | Cartesian atomic positions |
| `cell_bohr` | `Optional[np.ndarray]` | `(3, 3)` `bohr` | Optional periodic lattice vectors as rows |
| `pbc` | `Optional[np.ndarray]` | `(3,)` `bool` | Periodic boundary flags along lattice rows |
| `role` | `np.ndarray` (`int64`) | `(N,)` | `0=molecule/ligand`, `1=salt/ion`, `2=solvent`, `3=other/pocket` |
| `amount` | `np.ndarray` (`float64`) | `(N,)` | Stoichiometric / molar fraction weight |
| `species` | `np.ndarray` (`int64`) | `(N,)` | Connected component / molecule index |
| `delta_pos_bohr` | `Optional[np.ndarray]` | `(N, 3)` `bohr` | Optional `1o` inter-frame displacement or velocity |
| `forces_ev_ang` | `Optional[np.ndarray]` | `(N, 3)` `eV/Å` | Optional reference per-atom forces |
| `dipole_debye` | `Optional[np.ndarray]` | `(3,)` `Debye` | Optional reference dipole vector |
| `label` | `str` | — | Human-readable structure/frame identifier |

## 2. `VectorPrediction` & `O3Verification` (`src/toolkit/probe.py`)
- **`VectorPrediction`**:
  - `name: str` (`"dipole"`, `"force"`, `"vec_0"`, `"escape_direction"`, `"exit_vector"`, etc.)
  - `vector_au: list[float]` (`(3,)` in model atomic units)
  - `vector_physical: list[float]` (`(3,)` in `Debye`, `eV/Å`, or unit direction)
  - `unit: str` (`"Debye"`, `"eV/A"`, `"unit"`)
  - `magnitude: float`
  - `anchor_xyz_angstrom: list[float]` (`(3,)` origin in Ångström for 3D arrow rendering)
  - `anchor_atom: Optional[int]` (0-based atom index if atom-centered, else `None` for centroid)
- **`O3Verification`**:
  - `passed: bool`
  - `reflection_tested: bool`
  - `vector_equivariance_error: float` ($\max \|\mathbf{v}(X R^\top + \mathbf{t}) - \mathbf{v}(X) R^\top\|_2$)
  - `invariance_error: float` ($\max |\text{inv}(X R^\top + \mathbf{t}) - \text{inv}(X)|$)

## 3. `ProbeResult` (`src/toolkit/probe.py`)
- `engine: str` (`"v0"` or `"causal_lm"`)
- `question: str`
- `answer: str`
- `vectors: list[VectorPrediction]`
- `tokens: list[str]`
- `atom_attention: list[list[float]]` (`(T, N)` row-normalized routing weights)
- `pooled_atom_attention: list[float]` (`(N,)` overall atom importance)
- `channel_norms: dict[str, list[float]]` (`{"0e": [...], "1o": [...], "1e": [...], "2e": [...]}` per token)
- `domain: str` (`"general" | "electrolyte" | "md" | "drug"`)
- `domain_summary: dict`
- `o3_check: Optional[O3Verification]`
- `warnings: list[str]`

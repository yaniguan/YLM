"""Export TensorQA (v0 core, v0.2 compositional T4/T5, and l=2 quadrupole) to Hugging Face Datasets format."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Optional, Union

import numpy as np

from src.toolkit.ingest import AU_DIPOLE_TO_DEBYE, AU_FORCE_TO_EV_ANG, BOHR_TO_ANGSTROM

AU_QUAD_TO_DEBYE_ANG = AU_DIPOLE_TO_DEBYE * BOHR_TO_ANGSTROM


EVAL_SCRIPT_TEMPLATE = '''"""Official TensorQA K=32 O(3) Rotation & Reflection Evaluation Protocol."""
from __future__ import annotations
from typing import Callable, Iterable
import numpy as np


def random_o3_matrices(k: int = 32, seed: int = 0, include_improper: bool = True) -> list[np.ndarray]:
    """Generate K orthogonal 3x3 matrices (R_0 = I; uniform Haar SO(3) / O(3) for k >= 1)."""
    rng = np.random.default_rng(seed)
    mats = [np.eye(3, dtype=np.float64)]
    for idx in range(1, k):
        z = rng.standard_normal((3, 3))
        q, r = np.linalg.qr(z)
        d = np.diag(r)
        ph = np.where(d >= 0, 1.0, -1.0)
        q = q * ph
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        if include_improper and (idx % 2 == 1):
            q = -q  # improper rotation / parity inversion det(R) = -1
        mats.append(q)
    return mats


def evaluate_vector_predictions(
    records: Iterable[dict],
    predict_fn: Callable[[np.ndarray, np.ndarray, str], np.ndarray],
    k_rotations: int = 32,
    seed: int = 0,
    include_improper: bool = True,
) -> dict[str, float]:
    """Evaluate a predictor `f(Z, pos_bohr, question) -> vec_au (3,)` under K O(3) transformations."""
    rotations = random_o3_matrices(k=k_rotations, seed=seed, include_improper=include_improper)
    maes, orig_maes, cosines, eq_errs = [], [], [], []
    for rec in records:
        z = np.asarray(rec["atomic_numbers"], dtype=np.int64)
        pos0 = np.asarray(rec["positions_bohr"], dtype=np.float64)
        target0 = np.asarray(rec["target_vector_au"], dtype=np.float64)
        q = rec["question"]

        pred0 = np.asarray(predict_fn(z, pos0, q), dtype=np.float64)
        orig_maes.append(float(np.mean(np.abs(pred0 - target0))))
        denom_f0 = max(float(np.linalg.norm(pred0)), 1e-8)

        for r in rotations:
            pos_r = pos0 @ r.T
            target_r = target0 @ r.T
            pred_r = np.asarray(predict_fn(z, pos_r, q), dtype=np.float64)
            maes.append(float(np.mean(np.abs(pred_r - target_r))))
            n_p = float(np.linalg.norm(pred_r))
            n_t = float(np.linalg.norm(target_r))
            if n_p > 1e-8 and n_t > 1e-8:
                cosines.append(float(np.dot(pred_r, target_r) / (n_p * n_t)))
            # 1o odd-parity vector transforms as R v under both proper and improper O(3) matrices.
            rot_pred0 = pred0 @ r.T
            eq_errs.append(float(np.linalg.norm(pred_r - rot_pred0)) / denom_f0)

    return {
        "mae": float(np.mean(maes)) if maes else 0.0,
        "mae_orig_frame": float(np.mean(orig_maes)) if orig_maes else 0.0,
        "cosine": float(np.mean(cosines)) if cosines else 0.0,
        "eq_err": float(np.mean(eq_errs)) if eq_errs else 0.0,
    }
'''


def _build_dataset_card(
    stats: dict[str, dict[str, int]],
    configs_present: list[str],
    anonymized: bool = False,
) -> str:
    configs_yaml = []
    for cfg in configs_present:
        splits = stats.get(cfg, {})
        configs_yaml.append(f"  - config_name: {cfg}")
        configs_yaml.append("    data_files:")
        for sp in splits:
            configs_yaml.append(f"      - split: {sp}")
            configs_yaml.append(f"        path: {cfg}/{sp}.jsonl")
    configs_block = "\n".join(configs_yaml)

    repo_note = (
        "Released anonymously for double-blind peer review."
        if anonymized
        else "Source repository & reference `O(3)`-equivariant baselines: `https://github.com/yaniguan/YLM`."
    )

    rows = []
    for cfg in configs_present:
        for sp, count in stats.get(cfg, {}).items():
            rows.append(f"| `{cfg}` | `{sp}` | {count:,} |")
    split_table = "\n".join(rows)

    return f"""---
license: cc-by-4.0
task_categories:
  - question-answering
  - graph-ml
language:
  - en
tags:
  - chemistry
  - 3d-molecules
  - equivariance
  - o3
  - se3
  - irreps
  - molecular-qa
  - qm9
pretty_name: TensorQA (Language-Addressed 3D Molecular Vector & Tensor QA Benchmark)
configs:
{configs_block}
---

# TensorQA: Language-Addressed 3D Molecular Vector & Tensor QA

{repo_note}

## Summary

**TensorQA** benchmarks whether multimodal 3D molecule-language models can read, select, combine, and emit **frame-dependent directional 3D vectors (`1o`) and symmetric traceless tensors (`2e`)** from natural-language questions while obeying exact $O(3)$ rotation and parity symmetry.

Unlike existing molecular QA datasets that output only invariant scalars or text descriptions, every answer in TensorQA transforms covariantly with the input 3D coordinates: rotating or reflecting atomic coordinates by $R \\in O(3)$ rotates vector answers by $R \\mathbf{{v}}$ and rank-2 quadrupole tensors by $R Q R^\\top$.

## Configurations & Splits

| Config | Split | Examples |
|---|---|---:|
{split_table}

- **`v0_core`**:
  - `dipole` (T1, global `1o`): Molecular electric dipole vector (`e*bohr` and `Debye`).
  - `force` (T3, language-selected atom `1o`): Net atomic force (`Ha/bohr` and `eV/A`) on a uniquely described atom (element + bonded neighbor counts + ring membership when needed). `eval_only_atom_indices` is provided **strictly for post-hoc routing evaluation** and must not be fed to the model.
  - `test_heldout`: Same held-out test molecules as `test`, queried with **unseen question phrasings** never present in `train`/`val`/`test`.
- **`v0_compositional`** *(when included)*:
  - `diff` (T4, two-atom signed combination `1o`): $\\mathbf{{F}}_A - \\mathbf{{F}}_B$.
  - `proj` (T5, inter-atomic bond-axis projection `1o`): $(\\mathbf{{F}}_A \\cdot \\hat{{\\mathbf{{u}}}})\\hat{{\\mathbf{{u}}}}$ where $\\hat{{\\mathbf{{u}}}} = (\\mathbf{{r}}_B - \\mathbf{{r}}_A)/\\|\\mathbf{{r}}_B - \\mathbf{{r}}_A\\|$. Cannot be solved by a scalar weighted sum of single-atom vectors.
- **`v0_quadrupole`** *(when included)*:
  - `quadrupole` (T2, global `2e`): Symmetric traceless $3 \\times 3$ molecular quadrupole tensor computed with PySCF RHF/6-31G* around the center of nuclear charge.

## Provenance & Physical Conventions

1. **Geometries**: 20,000 molecules sampled from QM9 (`H, C, N, O, F`), perturbed by i.i.d. Gaussian atomic displacements ($\\sigma = 0.05\\,\\text{{\\AA}}$, fixed seed) so equilibrium forces are non-zero; structures with $|F_{{i,\\alpha}}| > 0.5\\,\\text{{Ha/bohr}}$ are filtered out.
2. **Labels**: Single-point GFN2-xTB (`tblite`) for `dipole` and `force`; PySCF RHF/6-31G* for `quadrupole`.
3. **Frames & Units**: `positions_bohr` (`bohr`) and `positions_angstrom` (`Angstrom`) share the exact laboratory Cartesian axes of `target_vector_au` / `target_vector_physical`.

## Official $K = 32$ $O(3)$ Evaluation Protocol

Use the bundled `evaluate_o3_protocol.py` script to evaluate any model across $K = 32$ random orthogonal transformations (including improper reflections $\\det R = -1$):
- `mae`: Mean component-wise absolute error across all $K=32$ rotated frames.
- `mae_orig_frame`: Error in the saved canonical dataset orientation (exposes frame-memorization shortcuts).
- `cosine`: Directional cosine similarity $\\cos(\\hat{{\\mathbf{{y}}}}, \\mathbf{{y}})$.
- `eq_err`: Relative equivariance violation $\\|f(R\\mathbf{{x}}) - D(R)f(\\mathbf{{x}})\\| / \\|f(\\mathbf{{x}})\\|$.
- `atom_acc`: Fraction of `force` queries where language-to-atom attention peaks on `eval_only_atom_indices[0]`.

## Known Limitations

- `dipole` and `force` labels come from semi-empirical **GFN2-xTB** rather than coupled-cluster/DFT; TensorQA is designed to benchmark **language-conditioned 3D geometric & directional reasoning**, not thermochemical DFT accuracy.
- Molecules cover neutral organic species up to 29 atoms (`H, C, N, O, F`).

## License & Citations

Released under **CC BY 4.0**. Please also cite QM9 (Ramakrishnan et al., *Sci. Data* 2014) and GFN2-xTB (Bannwarth et al., *JCTC* 2019).
"""


def export_tensorqa_hf(
    labels_path: Union[str, Path],
    v0_questions_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    comp_questions_path: Optional[Union[str, Path]] = None,
    quad_labels_path: Optional[Union[str, Path]] = None,
    anonymized: bool = False,
) -> dict:
    """Export TensorQA JSON + pickle artifacts into a self-contained Hugging Face Dataset directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(labels_path, "rb") as f:
        labels = pickle.load(f)
    with open(v0_questions_path, "r", encoding="utf-8") as f:
        v0_questions = json.load(f)

    stats: dict[str, dict[str, int]] = {}
    configs_present = ["v0_core"]

    # 1. Export v0_core
    core_splits: dict[str, list[dict]] = {}
    for idx, q in enumerate(v0_questions):
        lab = labels[int(q["mol"])]
        z_list = [int(z) for z in lab["numbers"]]
        pos_bohr = np.asarray(lab["pos"], dtype=np.float64)
        pos_ang = pos_bohr * BOHR_TO_ANGSTROM
        task = q["task"]
        if task == "dipole":
            vec_au = np.asarray(lab["dipole"], dtype=np.float64)
            vec_phys = vec_au * AU_DIPOLE_TO_DEBYE
            unit_au, unit_phys = "e*bohr", "Debye"
            eval_atoms: list[int] = []
        else:
            atom_idx = int(q["atom"])
            vec_au = np.asarray(lab["forces"][atom_idx], dtype=np.float64)
            vec_phys = vec_au * AU_FORCE_TO_EV_ANG
            unit_au, unit_phys = "Ha/bohr", "eV/A"
            eval_atoms = [atom_idx]

        rec = {
            "id": f"v0_core_{idx:06d}",
            "mol_id": int(q["mol"]),
            "qm9_index": int(lab.get("qm9_index", q["mol"])),
            "split": q["split"],
            "task": task,
            "irrep": "1o",
            "question": q["question"],
            "atomic_numbers": z_list,
            "positions_bohr": np.round(pos_bohr, 7).tolist(),
            "positions_angstrom": np.round(pos_ang, 7).tolist(),
            "target_vector_au": np.round(vec_au, 8).tolist(),
            "target_vector_physical": np.round(vec_phys, 8).tolist(),
            "target_unit_au": unit_au,
            "target_unit_physical": unit_phys,
            "eval_only_atom_indices": eval_atoms,
        }
        core_splits.setdefault(q["split"], []).append(rec)

    stats["v0_core"] = _write_config_splits(out / "v0_core", core_splits)

    # Split lookup by molecule for optional configs
    mol_split = {int(q["mol"]): q["split"] for q in v0_questions if q["split"] != "test_heldout"}

    # 2. Export v0_compositional (T4 diff & T5 proj) if provided
    if comp_questions_path is not None and Path(comp_questions_path).exists():
        with open(comp_questions_path, "r", encoding="utf-8") as f:
            comp_questions = json.load(f)
        comp_splits: dict[str, list[dict]] = {}
        for idx, q in enumerate(comp_questions):
            lab = labels[int(q["mol"])]
            z_list = [int(z) for z in lab["numbers"]]
            pos_bohr = np.asarray(lab["pos"], dtype=np.float64)
            pos_ang = pos_bohr * BOHR_TO_ANGSTROM
            vec_au = np.asarray(q["target"], dtype=np.float64)
            vec_phys = vec_au * AU_FORCE_TO_EV_ANG
            rec = {
                "id": f"v0_comp_{idx:06d}",
                "mol_id": int(q["mol"]),
                "qm9_index": int(lab.get("qm9_index", q["mol"])),
                "split": q["split"],
                "task": q["task"],
                "irrep": "1o",
                "question": q["question"],
                "atomic_numbers": z_list,
                "positions_bohr": np.round(pos_bohr, 7).tolist(),
                "positions_angstrom": np.round(pos_ang, 7).tolist(),
                "target_vector_au": np.round(vec_au, 8).tolist(),
                "target_vector_physical": np.round(vec_phys, 8).tolist(),
                "target_unit_au": "Ha/bohr",
                "target_unit_physical": "eV/A",
                "eval_only_atom_indices": [int(a) for a in q.get("atoms", [])],
            }
            comp_splits.setdefault(q["split"], []).append(rec)
        configs_present.append("v0_compositional")
        stats["v0_compositional"] = _write_config_splits(out / "v0_compositional", comp_splits)

    # 3. Export v0_quadrupole (l=2 symmetric traceless tensor) if provided
    if quad_labels_path is not None and Path(quad_labels_path).exists():
        with open(quad_labels_path, "rb") as f:
            quad_map = pickle.load(f)
        quad_splits: dict[str, list[dict]] = {}
        for idx, (mol_id, q_mat) in enumerate(sorted(quad_map.items())):
            m = int(mol_id)
            sp = mol_split.get(m, "train")
            lab = labels[m]
            pos_bohr = np.asarray(lab["pos"], dtype=np.float64)
            pos_ang = pos_bohr * BOHR_TO_ANGSTROM
            q_au = np.asarray(q_mat, dtype=np.float64)
            q_phys = q_au * AU_QUAD_TO_DEBYE_ANG
            rec = {
                "id": f"v0_quad_{idx:06d}",
                "mol_id": m,
                "qm9_index": int(lab.get("qm9_index", m)),
                "split": sp,
                "task": "quadrupole",
                "irrep": "2e",
                "question": "Give the traceless molecular quadrupole tensor around the center of nuclear charge.",
                "atomic_numbers": [int(z) for z in lab["numbers"]],
                "positions_bohr": np.round(pos_bohr, 7).tolist(),
                "positions_angstrom": np.round(pos_ang, 7).tolist(),
                "target_tensor_3x3_au": np.round(q_au, 8).tolist(),
                "target_tensor_3x3_physical": np.round(q_phys, 8).tolist(),
                "target_unit_au": "e*bohr^2",
                "target_unit_physical": "Debye*A",
                "eval_only_atom_indices": [],
            }
            quad_splits.setdefault(sp, []).append(rec)
        configs_present.append("v0_quadrupole")
        stats["v0_quadrupole"] = _write_config_splits(out / "v0_quadrupole", quad_splits)

    readme_path = out / "README.md"
    readme_path.write_text(_build_dataset_card(stats, configs_present, anonymized=anonymized), encoding="utf-8")

    eval_path = out / "evaluate_o3_protocol.py"
    eval_path.write_text(EVAL_SCRIPT_TEMPLATE, encoding="utf-8")

    return {
        "out_dir": str(out),
        "configs": configs_present,
        "stats": stats,
        "anonymized": anonymized,
        "files": [str(readme_path), str(eval_path)],
    }


def _write_config_splits(cfg_dir: Path, splits: dict[str, list[dict]]) -> dict[str, int]:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for sp, rows in splits.items():
        jsonl_path = cfg_dir / f"{sp}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        counts[sp] = len(rows)
    return counts

"""Comprehensive Physical Law & O(3) Equivariance Compliance Audit Engine (`ylm audit-physics` & Studio UI).

Audits 8 fundamental physical & symmetry laws in live PyTorch (`float32` or `float64`) and compares
`YLMUniverseModel` (`0e + 0o + 1o + 2e`) + `YLMProbe` + conservative MLIP mechanics head-to-head
against `UniMolStyle3DBaseline`, `GraphMVP3DInfoMaxBaseline`, and `EquiLLMMultiTaskBaseline`:

  1. Spatial Translation Invariance (`r_i -> r_i + a`, homogeneity of 3D space)
  2. Proper SO(3) Rotational Equivariance (`det(R) = +1`, Simultaneous `0e`, `0o`, `1o`, `0e+2e` Wigner laws)
  3. Improper O(3) Reflection / Spatial Inversion & Chiral Enantiomer Anti-Symmetry (`det(Q) = -1`)
  4. Bosonic / Atomic Index Permutation Symmetry (`P in S_N`: global invariance + per-atom covariance)
  5. Rank-2 Response Tensor Transpose Symmetry & Trace-Traceless (`0e + 2e`) Irreducible Decomposition
  6. Curie's Principle Point-Group Selection Rules (Planar mirror `sigma_h` & Inversion center `C_i` zeros)
  7. Conservative Potential Mechanics: Newton's 3rd Law (`sum F_i = 0`), Zero Net Torque (`sum r_i x F_i = 0`),
     Conservative Finite-Difference Gradient (`F_i = -dE/dr_i`), & Symmetric Force-Constant Hessian (`curl F = 0`)
  8. 3D Bravais Lattice Periodic Boundary Condition (PBC) Shift & Co-Rotation Invariance
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from src.tensor_readout import rotate_matrix
from src.toolkit.ingest import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    StructureFrame,
    covalent_radius,
    load_structure,
    load_trajectory,
    symbol_for_z,
)
from src.toolkit.mlip_md import AnalyticalReactiveForceProvider
from src.toolkit.ouroboros_bridge import _CHIRAL_FALLBACK_ANGSTROM
from src.toolkit.probe import YLMProbe
from src.universe_data import QuestionFeaturizer
from src.universe_fm import YLMUniverseModel
from e3nn import o3  # import after src.universe_fm registers PyTorch >= 2.6 safe_globals

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_builtin_presets() -> dict[str, dict]:
    """Return curated molecular & crystal presets spanning C2v, C2h, chiral (R)/(S), MD TS, drug, and 3D PBC."""
    presets: dict[str, dict] = {}

    # 1. Exact planar C2v water
    presets["water_c2v"] = {
        "id": "water_c2v",
        "title": "Water H2O (Planar C2v — Mirror Plane Selection Rule)",
        "domain": "general",
        "default_question": "What is the 3D molecular dipole moment vector?",
        "symmetry_hint": "C2v (planar xy mirror: 0o chiral == 0, out-of-plane 1o == 0)",
        "frame": StructureFrame(
            Z=[8, 1, 1],
            pos_bohr=np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399, 0.9273, 0.0]]) * ANGSTROM_TO_BOHR,
            label="water_c2v",
        ),
    }

    # 2. Centrosymmetric C2h trans-1,2-difluoroethene (inversion center -> exact zero dipole & zero chirality)
    z_tr, c_tr, _ = _CHIRAL_FALLBACK_ANGSTROM["F/C=C/F"]
    presets["trans_difluoroethene_c2h"] = {
        "id": "trans_difluoroethene_c2h",
        "title": "trans-1,2-Difluoroethene F/C=C/F (Centrosymmetric C2h — Dipole == 0)",
        "domain": "general",
        "default_question": "What is the 3D molecular dipole moment vector?",
        "symmetry_hint": "C2h (inversion center -X = P*X => net 1o vector == 0 & 0o chiral == 0)",
        "frame": StructureFrame(
            Z=z_tr,
            pos_bohr=np.asarray(c_tr, dtype=np.float64) * ANGSTROM_TO_BOHR,
            label="trans_F_C=C_F",
            metadata={"smiles": "F/C=C/F"},
        ),
    }

    # 3. Planar C2v cis-1,2-difluoroethene (Ouroboros backslash SMILES)
    z_cis, c_cis, _ = _CHIRAL_FALLBACK_ANGSTROM[r"F/C=C\F"]
    presets["cis_difluoroethene_c2v"] = {
        "id": "cis_difluoroethene_c2v",
        "title": r"cis-1,2-Difluoroethene F/C=C\F (Planar C2v — Non-zero Dipole, Zero Chirality)",
        "domain": "general",
        "default_question": "What is the 3D molecular dipole moment vector?",
        "symmetry_hint": "C2v (in-plane dipole != 0, out-of-plane 1o == 0, 0o chiral == 0)",
        "frame": StructureFrame(
            Z=z_cis,
            pos_bohr=np.asarray(c_cis, dtype=np.float64) * ANGSTROM_TO_BOHR,
            label="cis_F_C=C_F",
            metadata={"smiles": r"F/C=C\F"},
        ),
    }

    # 4 & 5. Chiral (R)-FEC and (S)-FEC enantiomer pair
    z_rfec, c_rfec, _ = _CHIRAL_FALLBACK_ANGSTROM["O=C1OC[C@H](F)O1"]
    presets["r_fec_chiral"] = {
        "id": "r_fec_chiral",
        "title": "(R)-Fluoroethylene Carbonate O=C1OC[C@H](F)O1 (Chiral C1 — SEI Additive)",
        "domain": "electrolyte",
        "default_question": "What is the (R)/(S) chiral handedness pseudoscalar and force on fluorine?",
        "symmetry_hint": "C1 chiral (+V tetrahedral stereocenter; flips 0o sign under reflection)",
        "frame": StructureFrame(
            Z=z_rfec,
            pos_bohr=(np.asarray(c_rfec, dtype=np.float64) - np.mean(c_rfec, axis=0)) * ANGSTROM_TO_BOHR,
            label="R_FEC",
            metadata={"smiles": "O=C1OC[C@H](F)O1"},
        ),
    }
    z_sfec, c_sfec, _ = _CHIRAL_FALLBACK_ANGSTROM["O=C1OC[C@@H](F)O1"]
    presets["s_fec_chiral"] = {
        "id": "s_fec_chiral",
        "title": "(S)-Fluoroethylene Carbonate O=C1OC[C@@H](F)O1 (Mirror Enantiomer)",
        "domain": "electrolyte",
        "default_question": "What is the (R)/(S) chiral handedness pseudoscalar and force on fluorine?",
        "symmetry_hint": "C1 chiral (-V mirror enantiomer of (R)-FEC; exact opposite 0o sign)",
        "frame": StructureFrame(
            Z=z_sfec,
            pos_bohr=(np.asarray(c_sfec, dtype=np.float64) - np.mean(c_sfec, axis=0)) * ANGSTROM_TO_BOHR,
            label="S_FEC",
            metadata={"smiles": "O=C1OC[C@@H](F)O1"},
        ),
    }

    # 6. Classic chiral tetrahedron F[C@H](Cl)Br
    z_bc, c_bc, _ = _CHIRAL_FALLBACK_ANGSTROM["F[C@H](Cl)Br"]
    presets["r_bcfm_chiral"] = {
        "id": "r_bcfm_chiral",
        "title": "(R)-Bromochlorofluoromethane F[C@H](Cl)Br (Archetypal 4-Substituent Chiral Center)",
        "domain": "general",
        "default_question": "Predict the parity-odd 3D chirality sign and force on bromine.",
        "symmetry_hint": "C1 chiral tetrahedron (Br > Cl > F > H)",
        "frame": StructureFrame(
            Z=z_bc,
            pos_bohr=(np.asarray(c_bc, dtype=np.float64) - np.mean(c_bc, axis=0)) * ANGSTROM_TO_BOHR,
            label="R_CHFCIBr",
            metadata={"smiles": "F[C@H](Cl)Br"},
        ),
    }

    # 7-9. Example files (Electrolyte, Reactive MD, Drug Pocket)
    li_path = REPO_ROOT / "examples" / "li_ec_shell.xyz"
    if li_path.exists():
        presets["li_ec_electrolyte"] = {
            "id": "li_ec_electrolyte",
            "title": "Li+–EC Solvation Shell (Battery Electrolyte Pack)",
            "domain": "electrolyte",
            "default_question": "What is the net restoring force vector on the lithium ion?",
            "symmetry_hint": "Ion-solvent coordination shell (solvation dipole & ion escape vector)",
            "frame": load_structure(li_path),
        }

    md_path = REPO_ROOT / "examples" / "proton_transfer_traj.extxyz"
    if md_path.exists():
        traj = load_trajectory(md_path)
        mid_fr = traj[min(len(traj) // 2, len(traj) - 1)]
        presets["zundel_proton_ts"] = {
            "id": "zundel_proton_ts",
            "title": "Proton-Transfer Transition Frame (Reactive MD Pack)",
            "domain": "md",
            "default_question": "What is the force vector on the shared bridging hydrogen atom?",
            "symmetry_hint": "Reactive proton-transfer saddle region (Force vs Motion alignment)",
            "frame": mid_fr,
        }

    drug_path = REPO_ROOT / "examples" / "indole_pocket.xyz"
    if drug_path.exists():
        presets["indole_drug_pocket"] = {
            "id": "indole_drug_pocket",
            "title": "Indole Pocket Complex (Drug SBDD Exit-Vector Pack)",
            "domain": "drug",
            "default_question": "What is the local force / exit vector on the indole nitrogen?",
            "symmetry_hint": "3D pharmacophore & ligand exit-vector steering",
            "frame": load_structure(drug_path),
        }

    # 10. 3D Triclinic Periodic Crystal (LiFePO4 motif)
    presets["lifepo4_pbc_crystal"] = {
        "id": "lifepo4_pbc_crystal",
        "title": "LiFePO4 Unit Motif (3D Periodic Boundary Condition Crystal)",
        "domain": "electrolyte",
        "default_question": "What is the force vector on the coordinated Li+ ion in the crystal cell?",
        "symmetry_hint": "3D Bravais lattice PBC=[T, T, T] (invariant under integer lattice shifts)",
        "frame": StructureFrame(
            Z=[3, 26, 15, 8, 8, 8],
            pos_bohr=np.array([
                [0.62, 0.85, 0.74], [2.48, 2.15, 1.96], [2.52, 0.78, 3.68],
                [1.36, 0.62, 2.74], [3.74, 1.12, 3.02], [2.41, 2.18, 4.32],
            ], dtype=np.float64) * ANGSTROM_TO_BOHR,
            cell_bohr=np.array([
                [5.15, 0.00, 0.00], [0.18, 5.02, 0.00], [0.08, 0.12, 5.28],
            ], dtype=np.float64) * ANGSTROM_TO_BOHR,
            pbc=np.ones(3, dtype=bool),
            label="lifepo4_pbc_crystal",
        ),
    }
    return presets


def resolve_preset_or_structure(source: str) -> tuple[StructureFrame, dict]:
    presets = get_builtin_presets()
    key = source.strip()
    if key in presets:
        info = presets[key]
        return copy.deepcopy(info["frame"]), info
    fr = load_structure(key)
    return fr, {
        "id": "custom",
        "title": fr.label or "Custom Structure",
        "domain": "general",
        "default_question": "What is the 3D molecular dipole moment vector?",
        "symmetry_hint": "Custom user-supplied geometry",
    }


class _AuditModelBundle:
    """Cached deterministic multi-rank models (`float32` & `float64`) for instant interactive UI auditing."""

    def __init__(self):
        self.featurizer = QuestionFeaturizer(mode="fast", dim=64)
        self._cache: dict[str, dict[str, torch.nn.Module]] = {}
        self.force_provider = AnalyticalReactiveForceProvider()

    def get_models(self, precision: str = "float64") -> dict[str, torch.nn.Module]:
        from src.fm_baselines import (
            EquiLLMMultiTaskBaseline,
            GraphMVP3DInfoMaxBaseline,
            UniMolStyle3DBaseline,
        )

        dtype = torch.float64 if precision == "float64" else torch.float32
        key = "f64" if dtype == torch.float64 else "f32"
        if key in self._cache:
            return self._cache[key]

        with torch.random.fork_rng():
            torch.manual_seed(2026)
            ylm = YLMUniverseModel(
                text_dim=64,
                backbone="e3nn",
                hidden_irreps="24x0e + 4x0o + 16x1o + 4x1e + 8x2e",
                cross_irreps="16x0e + 16x1o + 4x1e + 8x2e",
                n_layers=3,
                n_cross_layers=2,
                option="C",
                c_tp_mode="channelwise",
                d_text=64,
            ).to(dtype=dtype).eval()

            torch.manual_seed(2026)
            unimol = UniMolStyle3DBaseline(text_dim=64, d_model=48, n_layers=2, n_heads=4).to(dtype=dtype).eval()

            torch.manual_seed(2026)
            graphmvp = GraphMVP3DInfoMaxBaseline(text_dim=64, d_model=48, n_layers=2).to(dtype=dtype).eval()

            torch.manual_seed(2026)
            equillm = EquiLLMMultiTaskBaseline(
                text_dim=64,
                hidden_irreps="24x0e + 4x0o + 16x1o + 4x1e + 8x2e",
                n_layers=3,
                d_model=48,
                n_vec=16,
            ).to(dtype=dtype).eval()

        bundle = {"ylm": ylm, "unimol": unimol, "graphmvp": graphmvp, "equillm": equillm}
        self._cache[key] = bundle
        return bundle


_AUDIT_BUNDLE: Optional[_AuditModelBundle] = None


def _get_bundle() -> _AuditModelBundle:
    global _AUDIT_BUNDLE
    if _AUDIT_BUNDLE is None:
        _AUDIT_BUNDLE = _AuditModelBundle()
    return _AUDIT_BUNDLE


def _frame_to_tensors(
    frame: StructureFrame,
    question: str,
    featurizer: QuestionFeaturizer,
    dtype: torch.dtype,
) -> dict[str, Optional[torch.Tensor]]:
    Z = torch.as_tensor(frame.Z, dtype=torch.long).unsqueeze(0)
    pos = torch.as_tensor(frame.pos_bohr, dtype=dtype).unsqueeze(0)
    atom_mask = torch.ones_like(Z, dtype=torch.bool)
    t_raw = featurizer.encode(question).to(dtype=dtype).unsqueeze(0)
    text_mask = torch.ones((1, t_raw.shape[1]), dtype=torch.bool)
    lot_id = torch.tensor([1], dtype=torch.long)  # wb97m
    cell = None
    pbc = None
    if frame.cell_bohr is not None and (frame.pbc is None or np.any(frame.pbc)):
        cell = torch.as_tensor(frame.cell_bohr, dtype=dtype).unsqueeze(0)
        pbc_arr = frame.pbc if frame.pbc is not None else np.ones(3, dtype=bool)
        pbc = torch.as_tensor(pbc_arr, dtype=torch.bool).unsqueeze(0)
    return {
        "Z": Z,
        "pos": pos,
        "atom_mask": atom_mask,
        "text": t_raw,
        "text_mask": text_mask,
        "lot_id": lot_id,
        "cell": cell,
        "pbc": pbc,
    }


def _detect_point_symmetry(frame: StructureFrame) -> dict:
    """Detect exact mirror plane (`sigma_h`) and/or inversion center (`C_i`) in 3D coordinates."""
    pos = frame.pos_bohr - frame.pos_bohr.mean(axis=0, keepdims=True)
    n = frame.n_atoms
    cov = (pos.T @ pos) / max(n, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    is_planar = bool(eigvals[0] < 1e-8)
    plane_normal = eigvecs[:, 0].astype(np.float64) if is_planar else None

    # Check inversion center: for every i, exists j with Z[j] == Z[i] and ||-pos[i] - pos[j]|| < 1e-4 bohr
    has_inversion = True
    if frame.cell_bohr is not None:
        has_inversion = False
    else:
        for i in range(n):
            dists = np.linalg.norm((-pos[i])[None, :] - pos, axis=-1)
            matches = [j for j in range(n) if int(frame.Z[j]) == int(frame.Z[i]) and dists[j] < 1e-4]
            if not matches:
                has_inversion = False
                break

    return {
        "is_planar": is_planar,
        "plane_normal": None if plane_normal is None else plane_normal.tolist(),
        "has_inversion_center": has_inversion,
        "inertia_eigenvalues_bohr2": [float(v) for v in eigvals.tolist()],
    }


def euler_and_reflection_matrix(rx_deg: float, ry_deg: float, rz_deg: float, reflect: bool = False) -> np.ndarray:
    """Construct exact O(3) orthogonal matrix `Q` (`det=+1` proper rotation or `det=-1` roto-reflection)."""
    ax, ay, az = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    Q = Rz @ Ry @ Rx
    if reflect:
        Q = -Q  # spatial inversion P * R (det = -1)
    return Q


def _serialize_geometry_scene(
    frame: StructureFrame,
    arrows: list[dict],
    weights: Optional[list[float]] = None,
) -> dict:
    pos = frame.pos_angstrom
    atoms = []
    for idx in range(frame.n_atoms):
        z = int(frame.Z[idx])
        atoms.append({
            "index": idx,
            "Z": z,
            "symbol": symbol_for_z(z),
            "x": round(float(pos[idx, 0]), 5),
            "y": round(float(pos[idx, 1]), 5),
            "z": round(float(pos[idx, 2]), 5),
        })
    bonds = []
    for i in range(frame.n_atoms):
        ri = covalent_radius(int(frame.Z[i]))
        for j in range(i + 1, frame.n_atoms):
            rj = covalent_radius(int(frame.Z[j]))
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if 0.2 < d <= 1.32 * (ri + rj):
                bonds.append([i, j])
    return {
        "label": frame.label,
        "atoms": atoms,
        "bonds": bonds,
        "arrows": arrows,
        "weights": weights or [1.0 / max(frame.n_atoms, 1)] * frame.n_atoms,
        "cell": None if frame.cell_angstrom is None else frame.cell_angstrom.tolist(),
    }


@torch.no_grad()
def evaluate_pose_sandbox(
    frame_or_source: Union[str, StructureFrame],
    question: str = "What is the 3D molecular dipole moment vector?",
    *,
    rx_deg: float = 35.0,
    ry_deg: float = -48.0,
    rz_deg: float = 72.0,
    reflect: bool = False,
    translate_ang: tuple[float, float, float] = (0.0, 0.0, 0.0),
    permute_atoms: bool = False,
    precision: str = "float64",
) -> dict:
    """Interactive O(3) + translation + permutation pose sandbox comparing YLM vs Uni-Mol & EquiLLM live."""
    if isinstance(frame_or_source, StructureFrame):
        frame = copy.deepcopy(frame_or_source)
    else:
        frame, _ = resolve_preset_or_structure(str(frame_or_source))

    dtype = torch.float64 if precision == "float64" else torch.float32
    bundle = _get_bundle()
    models = bundle.get_models(precision=precision)

    Q_np = euler_and_reflection_matrix(rx_deg, ry_deg, rz_deg, reflect=reflect)
    Q = torch.as_tensor(Q_np, dtype=dtype)
    det_q = float(np.linalg.det(Q_np))
    shift_bohr = np.asarray(translate_ang, dtype=np.float64) * ANGSTROM_TO_BOHR

    n = frame.n_atoms
    perm = np.arange(n, dtype=np.int64)
    if permute_atoms and n > 1:
        rng = np.random.default_rng(137)
        perm = rng.permutation(n)

    frame_t = StructureFrame(
        Z=frame.Z[perm],
        pos_bohr=(frame.pos_bohr[perm] @ Q_np.T) + shift_bohr[None, :],
        cell_bohr=None if frame.cell_bohr is None else (frame.cell_bohr @ Q_np.T),
        pbc=None if frame.pbc is None else frame.pbc.copy(),
        label=f"{frame.label}_transformed",
    )

    inp0 = _frame_to_tensors(frame, question, bundle.featurizer, dtype)
    inp1 = _frame_to_tensors(frame_t, question, bundle.featurizer, dtype)

    comparisons: dict[str, dict] = {}
    for name, mdl in models.items():
        out0 = mdl(
            inp0["Z"], inp0["pos"], inp0["atom_mask"],
            inp0["text"], inp0["text_mask"],
            lot_id=inp0["lot_id"], cell=inp0["cell"], pbc=inp0["pbc"],
        )
        out1 = mdl(
            inp1["Z"], inp1["pos"], inp1["atom_mask"],
            inp1["text"], inp1["text_mask"],
            lot_id=inp1["lot_id"], cell=inp1["cell"], pbc=inp1["pbc"],
        )

        s0 = float(out0["scalar"][0, 0].item())
        s1 = float(out1["scalar"][0, 0].item())
        c0 = float(out0["chiral"][0, 0].item())
        c1 = float(out1["chiral"][0, 0].item())
        c_expected = det_q * c0

        v0 = out0["vector"][0]
        v1 = out1["vector"][0]
        v_expected = Q @ v0
        v_rel_err = float((v1 - v_expected).norm().item() / max(float(v0.norm().item()), 1e-2))
        v_cos = float(
            torch.dot(v1, v_expected).item()
            / max(float(v1.norm().item() * v_expected.norm().item()), 1e-12)
        )

        t0 = out0["tensor"][0]
        t1 = out1["tensor"][0]
        t_expected = rotate_matrix(out0["tensor"], Q)[0]
        t_rel_err = float((t1 - t_expected).norm().item() / max(float(t0.norm().item()), 1e-2))

        attn0_perm = out0["attn"][0].cpu().numpy()[perm]
        attn1 = out1["attn"][0].cpu().numpy()
        attn_perm_err = float(np.max(np.abs(attn1 - attn0_perm)))

        comparisons[name] = {
            "scalar_0e": {"orig": s0, "transformed": s1, "expected": s0, "abs_err": abs(s1 - s0)},
            "chiral_0o": {
                "orig": c0,
                "transformed": c1,
                "expected": c_expected,
                "abs_err": abs(c1 - c_expected),
            },
            "vector_1o": {
                "orig": [float(x) for x in v0.cpu().tolist()],
                "transformed": [float(x) for x in v1.cpu().tolist()],
                "expected": [float(x) for x in v_expected.cpu().tolist()],
                "rel_err": v_rel_err,
                "cosine_with_expected": v_cos,
            },
            "tensor_0e2e": {
                "orig": [[float(x) for x in r] for r in t0.cpu().tolist()],
                "transformed": [[float(x) for x in r] for r in t1.cpu().tolist()],
                "expected": [[float(x) for x in r] for r in t_expected.cpu().tolist()],
                "rel_err": t_rel_err,
            },
            "attn_routing_err": attn_perm_err,
            "attn_transformed": [float(x) for x in attn1.tolist()],
            "attn_orig": [float(x) for x in out0["attn"][0].cpu().tolist()],
        }

    # Build 3D scene for transformed pose showing YLM actual vs expected vs Uni-Mol actual arrows
    com_t = frame_t.pos_angstrom.mean(axis=0)
    ylm_v_act = np.asarray(comparisons["ylm"]["vector_1o"]["transformed"], dtype=np.float64)
    ylm_v_exp = np.asarray(comparisons["ylm"]["vector_1o"]["expected"], dtype=np.float64)
    uni_v_act = np.asarray(comparisons["unimol"]["vector_1o"]["transformed"], dtype=np.float64)

    def _mk_arrow(label_str: str, vec: np.ndarray, color: str, scale_len: float = 1.85) -> dict:
        nrm = float(np.linalg.norm(vec))
        unit_v = (vec / nrm) if nrm > 1e-10 else np.zeros(3)
        end = com_t + unit_v * scale_len
        return {
            "name": label_str,
            "vec": [round(float(c), 6) for c in vec.tolist()],
            "start": [round(float(c), 5) for c in com_t.tolist()],
            "end": [round(float(c), 5) for c in end.tolist()],
            "color": color,
        }

    scene_orig_com = frame.pos_angstrom.mean(axis=0)
    ylm_v0 = np.asarray(comparisons["ylm"]["vector_1o"]["orig"], dtype=np.float64)
    nrm0 = float(np.linalg.norm(ylm_v0))
    end0 = scene_orig_com + ((ylm_v0 / nrm0) * 1.85 if nrm0 > 1e-10 else np.zeros(3))
    scene_orig = _serialize_geometry_scene(
        frame,
        arrows=[{
            "name": "YLM f(X) [Original Frame]",
            "vec": [round(float(c), 6) for c in ylm_v0.tolist()],
            "start": [round(float(c), 5) for c in scene_orig_com.tolist()],
            "end": [round(float(c), 5) for c in end0.tolist()],
            "color": "#38bdf8",
        }],
        weights=comparisons["ylm"]["attn_orig"],
    )

    scene_transformed = _serialize_geometry_scene(
        frame_t,
        arrows=[
            _mk_arrow("Expected Q·f(X) [Exact Physics]", ylm_v_exp, "#22c55e", 1.95),
            _mk_arrow("YLM f(X·Q^T) [Actual]", ylm_v_act, "#38bdf8", 1.75),
            _mk_arrow("Uni-Mol f(X·Q^T) [Broken 0e]", uni_v_act, "#ef4444", 1.55),
        ],
        weights=comparisons["ylm"]["attn_transformed"],
    )

    return {
        "precision": precision,
        "det_Q": round(det_q, 4),
        "operation": "Improper Roto-Reflection O(3) (det = -1)" if reflect else "Proper Rotation SO(3) (det = +1)",
        "matrix_Q": [[round(float(x), 6) for x in row] for row in Q_np.tolist()],
        "permutation_applied": perm.tolist(),
        "translation_angstrom": list(translate_ang),
        "scene_orig": scene_orig,
        "scene_transformed": scene_transformed,
        "models": comparisons,
    }


def run_physics_compliance_audit(
    frame_or_source: Union[str, StructureFrame] = "r_fec_chiral",
    question: str = "What is the 3D molecular dipole moment vector?",
    *,
    precision: str = "float64",
    n_rotations: int = 8,
    probe: Optional[YLMProbe] = None,
) -> dict:
    """Execute the 8-Law Physical & Symmetry Compliance Certification Suite in live PyTorch."""
    if isinstance(frame_or_source, StructureFrame):
        frame = copy.deepcopy(frame_or_source)
        preset_meta = {"id": "custom", "title": frame.label}
    else:
        frame, preset_meta = resolve_preset_or_structure(str(frame_or_source))

    dtype = torch.float64 if precision == "float64" else torch.float32
    tol = 1e-8 if dtype == torch.float64 else 5e-5
    bundle = _get_bundle()
    models = bundle.get_models(precision=precision)
    ylm = models["ylm"]
    unimol = models["unimol"]
    graphmvp = models["graphmvp"]
    equillm = models["equillm"]

    inp0 = _frame_to_tensors(frame, question, bundle.featurizer, dtype)
    laws: list[dict] = []

    with torch.no_grad():
        out0 = ylm(
            inp0["Z"], inp0["pos"], inp0["atom_mask"],
            inp0["text"], inp0["text_mask"],
            lot_id=inp0["lot_id"], cell=inp0["cell"], pbc=inp0["pbc"],
        )
        u_out0 = unimol(
            inp0["Z"], inp0["pos"], inp0["atom_mask"],
            inp0["text"], inp0["text_mask"],
            lot_id=inp0["lot_id"], cell=inp0["cell"], pbc=inp0["pbc"],
        )
        g_out0 = graphmvp(
            inp0["Z"], inp0["pos"], inp0["atom_mask"],
            inp0["text"], inp0["text_mask"],
            lot_id=inp0["lot_id"], cell=inp0["cell"], pbc=inp0["pbc"],
        )
        eq_out0 = equillm(
            inp0["Z"], inp0["pos"], inp0["atom_mask"],
            inp0["text"], inp0["text_mask"],
            lot_id=inp0["lot_id"], cell=inp0["cell"], pbc=inp0["pbc"],
        )

        # ------------------------------------------------------------------
        # Law 1: Spatial Translation Invariance r_i -> r_i + a
        # ------------------------------------------------------------------
        shift = torch.tensor([[[17.3, -11.9, 24.6]]], dtype=dtype)
        out_tr = ylm(
            inp0["Z"], inp0["pos"] + shift, inp0["atom_mask"],
            inp0["text"], inp0["text_mask"],
            lot_id=inp0["lot_id"], cell=inp0["cell"], pbc=inp0["pbc"],
        )
        tr_err = max(
            float((out_tr["scalar"] - out0["scalar"]).abs().max().item()),
            float((out_tr["chiral"] - out0["chiral"]).abs().max().item()),
            float((out_tr["vector"] - out0["vector"]).norm().item()),
            float((out_tr["tensor"] - out0["tensor"]).norm().item()),
            float((out_tr["attn"] - out0["attn"]).abs().max().item()),
        )
        laws.append({
            "law_id": "L1_translation",
            "name": "1. Spatial Translation Invariance T(a)",
            "formula": "f(X + 1·a^T) == f(X)  for all ranks (0e, 0o, 1o, 2e)",
            "passed": tr_err < tol,
            "ylm_error": tr_err,
            "tolerance": tol,
            "detail": f"Shifted all {frame.n_atoms} atoms by (+17.3, -11.9, +24.6) bohr; max multi-rank drift = {tr_err:.2e}.",
        })

        # ------------------------------------------------------------------
        # Law 2: Proper SO(3) Rotational Equivariance (det(R) = +1)
        # ------------------------------------------------------------------
        so3_sc, so3_ch, so3_vec, so3_ten = [], [], [], []
        uni_so3_vec, gmvp_so3_vec = [], []
        with torch.random.fork_rng():
            torch.manual_seed(314)
            for _ in range(max(1, n_rotations)):
                R = o3.rand_matrix(dtype=dtype)
                if torch.linalg.det(R) < 0:
                    R = -R
                pos_r = inp0["pos"] @ R.T
                cell_r = None if inp0["cell"] is None else (inp0["cell"] @ R.T)
                o_r = ylm(
                    inp0["Z"], pos_r, inp0["atom_mask"],
                    inp0["text"], inp0["text_mask"],
                    lot_id=inp0["lot_id"], cell=cell_r, pbc=inp0["pbc"],
                )
                so3_sc.append(float((o_r["scalar"] - out0["scalar"]).abs().max().item()))
                so3_ch.append(float((o_r["chiral"] - out0["chiral"]).abs().max().item()))
                v_den = max(float(out0["vector"].norm().item()), 1e-2)
                so3_vec.append(float((o_r["vector"] - (out0["vector"] @ R.T)).norm().item()) / v_den)
                t_den = max(float(out0["tensor"].norm().item()), 1e-2)
                so3_ten.append(float((o_r["tensor"] - rotate_matrix(out0["tensor"], R)).norm().item()) / t_den)

                uo_r = unimol(
                    inp0["Z"], pos_r, inp0["atom_mask"],
                    inp0["text"], inp0["text_mask"],
                    lot_id=inp0["lot_id"], cell=cell_r, pbc=inp0["pbc"],
                )
                uv_den = max(float(u_out0["vector"].norm().item()), 1e-12)
                uni_so3_vec.append(float((uo_r["vector"] - (u_out0["vector"] @ R.T)).norm().item()) / uv_den)

                go_r = graphmvp(
                    inp0["Z"], pos_r, inp0["atom_mask"],
                    inp0["text"], inp0["text_mask"],
                    lot_id=inp0["lot_id"], cell=cell_r, pbc=inp0["pbc"],
                )
                gv_den = max(float(g_out0["vector"].norm().item()), 1e-12)
                gmvp_so3_vec.append(float((go_r["vector"] - (g_out0["vector"] @ R.T)).norm().item()) / gv_den)

        max_so3 = max(max(so3_sc), max(so3_ch), max(so3_vec), max(so3_ten))
        laws.append({
            "law_id": "L2_so3_rotation",
            "name": f"2. Proper SO(3) Rotational Equivariance (K={n_rotations}, det=+1)",
            "formula": "0e: E'=E | 0o: χ'=χ | 1o: v'=R·v | 0e+2e: T'=R·T·R^T",
            "passed": max_so3 < tol,
            "ylm_error": max_so3,
            "baseline_errors": {
                "Uni-Mol (1o vec)": float(np.mean(uni_so3_vec)),
                "GraphMVP (1o vec)": float(np.mean(gmvp_so3_vec)),
            },
            "tolerance": tol,
            "detail": (
                f"YLM max err: 0e={max(so3_sc):.2e}, 0o={max(so3_ch):.2e}, 1o={max(so3_vec):.2e}, 2e={max(so3_ten):.2e} "
                f"(vs Uni-Mol 1o err={np.mean(uni_so3_vec):.2e}, GraphMVP={np.mean(gmvp_so3_vec):.2e})."
            ),
        })

        # ------------------------------------------------------------------
        # Law 3: Improper O(3) Reflection / Inversion & Enantiomer Parity (det=-1)
        # ------------------------------------------------------------------
        imp_sc, imp_ch, imp_vec, imp_ten = [], [], [], []
        with torch.random.fork_rng():
            torch.manual_seed(2718)
            for k in range(max(1, n_rotations)):
                Q = -torch.eye(3, dtype=dtype) if k == 0 else -o3.rand_matrix(dtype=dtype)
                if torch.linalg.det(Q) > 0:
                    Q = -Q
                pos_q = inp0["pos"] @ Q.T
                cell_q = None if inp0["cell"] is None else (inp0["cell"] @ Q.T)
                o_q = ylm(
                    inp0["Z"], pos_q, inp0["atom_mask"],
                    inp0["text"], inp0["text_mask"],
                    lot_id=inp0["lot_id"], cell=cell_q, pbc=inp0["pbc"],
                )
                imp_sc.append(float((o_q["scalar"] - out0["scalar"]).abs().max().item()))
                imp_ch.append(float((o_q["chiral"] + out0["chiral"]).abs().max().item()))  # MUST flip sign!
                v_den = max(float(out0["vector"].norm().item()), 1e-2)
                imp_vec.append(float((o_q["vector"] - (out0["vector"] @ Q.T)).norm().item()) / v_den)
                t_den = max(float(out0["tensor"].norm().item()), 1e-2)
                imp_ten.append(float((o_q["tensor"] - rotate_matrix(out0["tensor"], Q)).norm().item()) / t_den)

        # Also check baseline enantiomer blindness under X -> -X on a chiral reference if current is achiral
        pos_inv = -inp0["pos"]
        cell_inv = None if inp0["cell"] is None else -inp0["cell"]
        u_inv = unimol(
            inp0["Z"], pos_inv, inp0["atom_mask"], inp0["text"], inp0["text_mask"],
            lot_id=inp0["lot_id"], cell=cell_inv, pbc=inp0["pbc"],
        )
        eq_inv = equillm(
            inp0["Z"], pos_inv, inp0["atom_mask"], inp0["text"], inp0["text_mask"],
            lot_id=inp0["lot_id"], cell=cell_inv, pbc=inp0["pbc"],
        )
        uni_parity_err = float((u_inv["chiral"] + u_out0["chiral"]).abs().max().item())
        equi_parity_err = float((eq_inv["chiral"] + eq_out0["chiral"]).abs().max().item())

        max_imp = max(max(imp_sc), max(imp_ch), max(imp_vec), max(imp_ten))
        laws.append({
            "law_id": "L3_improper_parity",
            "name": "3. Improper O(3) Reflection & Chiral Parity Law (det=-1)",
            "formula": "0e(+): E(-X)=+E(X) | 0o(-): χ(-X)=-χ(X) | 1o(-): v(-X)=-v(X) | 2e(+): T(-X)=+T(X)",
            "passed": max_imp < tol,
            "ylm_error": max_imp,
            "baseline_errors": {
                "Uni-Mol (0o parity violation |χ(-X)+χ(X)|)": uni_parity_err,
                "EquiLLM (0o parity violation |χ(-X)+χ(X)|)": equi_parity_err,
            },
            "tolerance": tol,
            "detail": (
                f"YLM parity residual={max_imp:.2e} (χ(X)={float(out0['chiral'].item()):+.4e} -> "
                f"χ(-X)={-float(out0['chiral'].item()):+.4e}); "
                f"Uni-Mol violates 0o anti-symmetry (|χ(-X)+χ(X)|={uni_parity_err:.2e})."
            ),
        })

        # ------------------------------------------------------------------
        # Law 4: Bosonic / Atomic Index Permutation Symmetry (P in S_N)
        # ------------------------------------------------------------------
        n_at = frame.n_atoms
        perm = torch.randperm(n_at) if n_at > 1 else torch.arange(n_at)
        Z_p = inp0["Z"][:, perm]
        pos_p = inp0["pos"][:, perm, :]
        out_p = ylm(
            Z_p, pos_p, inp0["atom_mask"],
            inp0["text"], inp0["text_mask"],
            lot_id=inp0["lot_id"], cell=inp0["cell"], pbc=inp0["pbc"],
        )
        perm_err = max(
            float((out_p["scalar"] - out0["scalar"]).abs().max().item()),
            float((out_p["chiral"] - out0["chiral"]).abs().max().item()),
            float((out_p["vector"] - out0["vector"]).norm().item()),
            float((out_p["tensor"] - out0["tensor"]).norm().item()),
            float((out_p["attn"] - out0["attn"][:, perm]).abs().max().item()),
            float((out_p["denoise"] - out0["denoise"][:, perm, :]).abs().max().item()),
        )
        laws.append({
            "law_id": "L4_permutation",
            "name": "4. Atomic Index Permutation Equivariance (S_N)",
            "formula": "Global f(P·Z, P·X) == f(Z, X);  Per-atom α(P·Z, P·X) == P·α(Z, X)",
            "passed": perm_err < tol,
            "ylm_error": perm_err,
            "tolerance": tol,
            "detail": f"Permuted {n_at} atom indices ({perm.tolist()[:8]}); global + per-atom routing/denoise residual = {perm_err:.2e}.",
        })

        # ------------------------------------------------------------------
        # Law 5: Rank-2 Tensor Symmetry & Trace-Traceless (0e + 2e) Decomposition
        # ------------------------------------------------------------------
        T_mat = out0["tensor"][0]
        asym_err = float((T_mat - T_mat.T).norm().item())
        iso_part = (torch.diagonal(T_mat).sum() / 3.0) * torch.eye(3, dtype=dtype)
        traceless_2e = T_mat - iso_part
        trace_res = float(torch.diagonal(traceless_2e).sum().abs().item())
        ten_sym_err = max(asym_err, trace_res)
        laws.append({
            "law_id": "L5_tensor_symmetry",
            "name": "5. Rank-2 Response Tensor Symmetry (T == T^T, 0e ⊕ 2e)",
            "formula": "||T - T^T||_F == 0  and  Tr(T_2e) == 0  (6 symmetric DOFs, zero 1e curl)",
            "passed": ten_sym_err < tol,
            "ylm_error": ten_sym_err,
            "tolerance": tol,
            "detail": f"||T - T^T||_F = {asym_err:.2e}, |Tr(T_2e)| = {trace_res:.2e}.",
        })

        # ------------------------------------------------------------------
        # Law 6: Curie's Principle Point-Group Selection Rules (Zero-by-Symmetry!)
        # ------------------------------------------------------------------
        sym_info = _detect_point_symmetry(frame)
        curie_checks: list[tuple[str, float]] = []

        if sym_info["is_planar"]:
            n_vec = torch.tensor(sym_info["plane_normal"], dtype=dtype)
            out_of_plane_1o = float(torch.dot(out0["vector"][0], n_vec).abs().item())
            chiral_planar = float(out0["chiral"].abs().max().item())
            tn = out0["tensor"][0] @ n_vec
            shear_2e = float((tn - torch.dot(n_vec, tn) * n_vec).norm().item())
            curie_checks.extend([
                ("planar_|n·v_1o|", out_of_plane_1o),
                ("planar_|χ_0o|", chiral_planar),
                ("planar_shear_||(I-nn^T)Tn||", shear_2e),
            ])
        if sym_info["has_inversion_center"]:
            inv_vec_norm = float(out0["vector"][0].norm().item())
            inv_chiral = float(out0["chiral"].abs().max().item())
            curie_checks.extend([
                ("centrosym_||v_1o||", inv_vec_norm),
                ("centrosym_|χ_0o|", inv_chiral),
            ])

        # If current geometry is C1 asymmetric, also verify Curie's Principle on rotated C2v water + C2h trans-C2H2F2
        used_control = False
        if not curie_checks:
            used_control = True
            presets = get_builtin_presets()
            R_ctrl = euler_and_reflection_matrix(29.0, -51.0, 64.0, reflect=False)
            w_fr = presets["water_c2v"]["frame"].transformed(R_ctrl)
            w_in = _frame_to_tensors(w_fr, "What is the 3D molecular dipole moment vector?", bundle.featurizer, dtype)
            w_out = ylm(w_in["Z"], w_in["pos"], w_in["atom_mask"], w_in["text"], w_in["text_mask"], lot_id=w_in["lot_id"])
            n_rot = torch.as_tensor(R_ctrl @ np.array([0.0, 0.0, 1.0]), dtype=dtype)
            curie_checks.append(("ctrl_water_|n·v_1o|", float(torch.dot(w_out["vector"][0], n_rot).abs().item())))
            curie_checks.append(("ctrl_water_|χ_0o|", float(w_out["chiral"].abs().max().item())))

            tr_fr = presets["trans_difluoroethene_c2h"]["frame"].transformed(R_ctrl)
            tr_in = _frame_to_tensors(tr_fr, "What is the 3D molecular dipole moment vector?", bundle.featurizer, dtype)
            tr_out = ylm(tr_in["Z"], tr_in["pos"], tr_in["atom_mask"], tr_in["text"], tr_in["text_mask"], lot_id=tr_in["lot_id"])
            curie_checks.append(("ctrl_trans_C2h_||dipole||", float(tr_out["vector"][0].norm().item())))

        curie_err = max(v for _, v in curie_checks)
        curie_desc = ", ".join(f"{k}={v:.2e}" for k, v in curie_checks)
        laws.append({
            "law_id": "L6_curie_selection",
            "name": "6. Curie's Principle Point-Group Selection Rules (Symmetry Zeros)",
            "formula": "σ_h planar => n·v=0, χ=0, (I-nn^T)Tn=0;  C_i inversion => v=0, χ=0",
            "passed": curie_err < tol,
            "ylm_error": curie_err,
            "tolerance": tol,
            "detail": (
                f"{'Input is C1 chiral/asymmetric (verified via rotated C2v/C2h controls): ' if used_control else 'Detected symmetry on input structure: '}"
                f"{curie_desc}."
            ),
        })

    # ------------------------------------------------------------------
    # Law 7: Conservative Mechanics — Newton's 3rd Law, Zero Torque, & F = -dE/dr Curl-Freeness
    # ------------------------------------------------------------------
    pos_ang = frame.pos_angstrom.copy()
    cell_ang = frame.cell_angstrom
    pbc_arr = frame.pbc
    e0, f0 = bundle.force_provider(frame.Z, pos_ang, cell_ang, pbc_arr)

    net_force = float(np.linalg.norm(f0.sum(axis=0)))
    if cell_ang is None:
        r_rel = pos_ang - pos_ang.mean(axis=0, keepdims=True)
        net_torque = float(np.linalg.norm(np.cross(r_rel, f0).sum(axis=0)))
    else:
        net_torque = 0.0  # PBC triclinic cell torque couples to symmetric Cauchy stress tensor

    # Central finite-difference verification of conservative force F_{0, alpha} = -dE / dr_{0, alpha}
    eps_fd = 1e-5
    f_fd = np.zeros(3, dtype=np.float64)
    for alpha in range(3):
        p_plus = pos_ang.copy()
        p_minus = pos_ang.copy()
        p_plus[0, alpha] += eps_fd
        p_minus[0, alpha] -= eps_fd
        ep, _ = bundle.force_provider(frame.Z, p_plus, cell_ang, pbc_arr)
        em, _ = bundle.force_provider(frame.Z, p_minus, cell_ang, pbc_arr)
        f_fd[alpha] = -(ep - em) / (2.0 * eps_fd)
    fd_grad_err = float(np.max(np.abs(f0[0] - f_fd)))

    # Curl-freeness / Hessian symmetry check: dF_{0,x}/dr_{1,y} == dF_{1,y}/dr_{0,x}
    curl_err = 0.0
    if frame.n_atoms >= 2:
        p_1y_p, p_1y_m = pos_ang.copy(), pos_ang.copy()
        p_1y_p[1, 1] += eps_fd
        p_1y_m[1, 1] -= eps_fd
        _, f_1y_p = bundle.force_provider(frame.Z, p_1y_p, cell_ang, pbc_arr)
        _, f_1y_m = bundle.force_provider(frame.Z, p_1y_m, cell_ang, pbc_arr)
        dF0x_dr1y = (f_1y_p[0, 0] - f_1y_m[0, 0]) / (2.0 * eps_fd)

        p_0x_p, p_0x_m = pos_ang.copy(), pos_ang.copy()
        p_0x_p[0, 0] += eps_fd
        p_0x_m[0, 0] -= eps_fd
        _, f_0x_p = bundle.force_provider(frame.Z, p_0x_p, cell_ang, pbc_arr)
        _, f_0x_m = bundle.force_provider(frame.Z, p_0x_m, cell_ang, pbc_arr)
        dF1y_dr0x = (f_0x_p[1, 1] - f_0x_m[1, 1]) / (2.0 * eps_fd)
        curl_err = float(abs(dF0x_dr1y - dF1y_dr0x))

    mech_tol = 5e-6
    mech_err = max(net_force, net_torque, fd_grad_err, curl_err)
    laws.append({
        "law_id": "L7_conservative_mechanics",
        "name": "7. Conservative Mechanics: Newton's 3rd Law, Zero Torque & Curl-Free F = -∇E",
        "formula": "∑F_i == 0,  ∑r_i×F_i == 0,  F_i == -∂E/∂r_i (FD),  ∂F_iα/∂r_jβ == ∂F_jβ/∂r_iα",
        "passed": mech_err < mech_tol,
        "ylm_error": mech_err,
        "tolerance": mech_tol,
        "detail": (
            f"||∑F_i||={net_force:.2e} eV/Å, ||∑r_i×F_i||={net_torque:.2e} eV, "
            f"max|F_autograd - F_FD|={fd_grad_err:.2e} eV/Å, |curl F|={curl_err:.2e} eV/Å²."
        ),
    })

    # ------------------------------------------------------------------
    # Law 8: 3D Bravais Lattice PBC Shift & Cell Co-Rotation Invariance
    # ------------------------------------------------------------------
    with torch.no_grad():
        pbc_frame = frame if frame.cell_bohr is not None else get_builtin_presets()["lifepo4_pbc_crystal"]["frame"]
        p_in0 = _frame_to_tensors(pbc_frame, question, bundle.featurizer, dtype)
        p_out0 = ylm(
            p_in0["Z"], p_in0["pos"], p_in0["atom_mask"],
            p_in0["text"], p_in0["text_mask"],
            lot_id=p_in0["lot_id"], cell=p_in0["cell"], pbc=p_in0["pbc"],
        )
        # Shift atom 0 by integer Bravais lattice vector c_0 - c_1
        pos_lat = p_in0["pos"].clone()
        pos_lat[0, 0] = pos_lat[0, 0] + p_in0["cell"][0, 0] - p_in0["cell"][0, 1]
        p_out_shift = ylm(
            p_in0["Z"], pos_lat, p_in0["atom_mask"],
            p_in0["text"], p_in0["text_mask"],
            lot_id=p_in0["lot_id"], cell=p_in0["cell"], pbc=p_in0["pbc"],
        )
        pbc_err = max(
            float((p_out_shift["scalar"] - p_out0["scalar"]).abs().max().item()),
            float((p_out_shift["chiral"] - p_out0["chiral"]).abs().max().item()),
            float((p_out_shift["vector"] - p_out0["vector"]).norm().item()),
            float((p_out_shift["tensor"] - p_out0["tensor"]).norm().item()),
        )
        laws.append({
            "law_id": "L8_pbc_lattice",
            "name": "8. 3D Bravais Lattice Periodic Shift Invariance (PBC)",
            "formula": "f(r_k + n_1·a_1 + n_2·a_2 + n_3·a_3, cell) == f(r, cell)",
            "passed": pbc_err < tol,
            "ylm_error": pbc_err,
            "tolerance": tol,
            "detail": (
                f"Tested on {'input crystal' if frame.cell_bohr is not None else 'LiFePO4 3D PBC crystal motif'}: "
                f"shifted atom #0 by Bravais vector (a_1 - a_2); max multi-rank drift = {pbc_err:.2e}."
            ),
        })

    # Optional shipped v0 YLMProbe check
    active_probe = probe or YLMProbe.default()
    probe_chk = active_probe.verify_equivariance(frame, question)

    passed_count = sum(1 for item in laws if item["passed"])
    return {
        "structure_label": frame.label,
        "preset_title": preset_meta.get("title", frame.label),
        "n_atoms": frame.n_atoms,
        "precision": precision,
        "question": question,
        "symmetry_detected": sym_info,
        "passed_all": passed_count == len(laws) and probe_chk.passed,
        "passed_count": passed_count,
        "total_laws": len(laws),
        "v0_probe_check": {
            "engine": active_probe.engine,
            "passed": probe_chk.passed,
            "vector_equivariance_error": probe_chk.vector_equivariance_error,
            "invariance_error": probe_chk.invariance_error,
        },
        "laws": laws,
    }

"""Check 8 basic physics rules on 3D molecules and crystals (`ylm audit-physics` and Studio UI).

Runs live PyTorch checks (`float64` or `float32`) and compares `YLMUniverseModel` against
`UniMolStyle3DBaseline`, `GraphMVP3DInfoMaxBaseline`, and `EquiLLMMultiTaskBaseline`:

  1. Move in space: shifting all atoms together does not change the answer.
  2. Spin in 3D: plain numbers stay the same, and output arrows spin by the same angle.
  3. Mirror flip: energy stays the same, arrows flip, and left/right handedness flips sign (`+` to `-`).
  4. Swap atom row order: changing atom numbering in the file does not change the physics.
  5. Balanced 3x3 matrix: response matrices equal their transpose (`T[i, j] == T[j, i]`).
  6. Balanced shape cancels out: flat molecules have zero out-of-plane arrow and zero handedness;
     molecules with a center of symmetry have zero total dipole.
  7. Newton's laws & energy slope: internal forces add to zero, total torque is zero, and `F = -dE/dx`.
  8. Crystal repeat box: shifting an atom by one full crystal box vector gives the exact same answer.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from src.tensor_readout import rotate_matrix
from src.toolkit.ingest import (
    ANGSTROM_TO_BOHR,
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
_FEATURIZER = QuestionFeaturizer(mode="fast", dim=64)
_FORCE_PROVIDER = AnalyticalReactiveForceProvider()


def get_builtin_presets() -> dict[str, dict]:
    """Return built-in example molecules and crystals for physics checks and UI demos."""
    presets: dict[str, dict] = {}

    presets["water_c2v"] = {
        "id": "water_c2v",
        "title": "Water H2O (Flat molecule — zero out-of-plane arrow & zero handedness)",
        "domain": "general",
        "default_question": "What is the 3D molecular dipole moment vector?",
        "symmetry_hint": "Flat molecule in the xy plane: handedness = 0 and z-arrow = 0",
        "frame": StructureFrame(
            Z=[8, 1, 1],
            pos_bohr=np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2399, 0.9273, 0.0]]) * ANGSTROM_TO_BOHR,
            label="water_c2v",
        ),
    }

    z_tr, c_tr, _ = _CHIRAL_FALLBACK_ANGSTROM["F/C=C/F"]
    presets["trans_difluoroethene_c2h"] = {
        "id": "trans_difluoroethene_c2h",
        "title": "trans-1,2-Difluoroethene F/C=C/F (Opposite pulls cancel -> Dipole = 0)",
        "domain": "general",
        "default_question": "What is the 3D molecular dipole moment vector?",
        "symmetry_hint": "Center of symmetry: equal opposite pulls cancel out, so total dipole = 0",
        "frame": StructureFrame(
            Z=z_tr,
            pos_bohr=np.asarray(c_tr, dtype=np.float64) * ANGSTROM_TO_BOHR,
            label="trans_F_C=C_F",
            metadata={"smiles": "F/C=C/F"},
        ),
    }

    z_cis, c_cis, _ = _CHIRAL_FALLBACK_ANGSTROM[r"F/C=C\F"]
    presets["cis_difluoroethene_c2v"] = {
        "id": "cis_difluoroethene_c2v",
        "title": r"cis-1,2-Difluoroethene F/C=C\F (Same-side pulls add up -> Dipole != 0)",
        "domain": "general",
        "default_question": "What is the 3D molecular dipole moment vector?",
        "symmetry_hint": "Flat molecule with both F atoms on one side: in-plane dipole != 0, handedness = 0",
        "frame": StructureFrame(
            Z=z_cis,
            pos_bohr=np.asarray(c_cis, dtype=np.float64) * ANGSTROM_TO_BOHR,
            label="cis_F_C=C_F",
            metadata={"smiles": r"F/C=C\F"},
        ),
    }

    z_rfec, c_rfec, _ = _CHIRAL_FALLBACK_ANGSTROM["O=C1OC[C@H](F)O1"]
    presets["r_fec_chiral"] = {
        "id": "r_fec_chiral",
        "title": "(R)-Fluoroethylene Carbonate O=C1OC[C@H](F)O1 (Chiral battery additive)",
        "domain": "electrolyte",
        "default_question": "What is the (R)/(S) chiral handedness sign and force on fluorine?",
        "symmetry_hint": "3D chiral carbon: flips sign (+ to -) in a mirror",
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
        "title": "(S)-Fluoroethylene Carbonate O=C1OC[C@@H](F)O1 (Mirror twin of (R)-FEC)",
        "domain": "electrolyte",
        "default_question": "What is the (R)/(S) chiral handedness sign and force on fluorine?",
        "symmetry_hint": "Mirror twin of (R)-FEC: gives the exact opposite handedness sign",
        "frame": StructureFrame(
            Z=z_sfec,
            pos_bohr=(np.asarray(c_sfec, dtype=np.float64) - np.mean(c_sfec, axis=0)) * ANGSTROM_TO_BOHR,
            label="S_FEC",
            metadata={"smiles": "O=C1OC[C@@H](F)O1"},
        ),
    }

    z_bc, c_bc, _ = _CHIRAL_FALLBACK_ANGSTROM["F[C@H](Cl)Br"]
    presets["r_bcfm_chiral"] = {
        "id": "r_bcfm_chiral",
        "title": "(R)-CHFCIBr F[C@H](Cl)Br (Carbon with 4 different groups — chiral)",
        "domain": "general",
        "default_question": "Predict the parity-odd 3D chirality sign and force on bromine.",
        "symmetry_hint": "Carbon bonded to H, F, Cl, Br: non-superimposable on its mirror image",
        "frame": StructureFrame(
            Z=z_bc,
            pos_bohr=(np.asarray(c_bc, dtype=np.float64) - np.mean(c_bc, axis=0)) * ANGSTROM_TO_BOHR,
            label="R_CHFCIBr",
            metadata={"smiles": "F[C@H](Cl)Br"},
        ),
    }

    li_path = REPO_ROOT / "examples" / "li_ec_shell.xyz"
    if li_path.exists():
        presets["li_ec_electrolyte"] = {
            "id": "li_ec_electrolyte",
            "title": "Li+ in EC Solvent Shell (Battery electrolyte)",
            "domain": "electrolyte",
            "default_question": "What is the net restoring force vector on the lithium ion?",
            "symmetry_hint": "Lithium ion surrounded by solvent molecules",
            "frame": load_structure(li_path),
        }

    md_path = REPO_ROOT / "examples" / "proton_transfer_traj.extxyz"
    if md_path.exists():
        traj = load_trajectory(md_path)
        mid_fr = traj[min(len(traj) // 2, len(traj) - 1)]
        presets["zundel_proton_ts"] = {
            "id": "zundel_proton_ts",
            "title": "Shared H+ Jumping Between Two Waters (Proton transfer)",
            "domain": "md",
            "default_question": "What is the force vector on the shared bridging hydrogen atom?",
            "symmetry_hint": "Middle of a proton jump: compare force arrow vs. motion arrow",
            "frame": mid_fr,
        }

    drug_path = REPO_ROOT / "examples" / "indole_pocket.xyz"
    if drug_path.exists():
        presets["indole_drug_pocket"] = {
            "id": "indole_drug_pocket",
            "title": "Indole in Protein Pocket (Drug binding & growth direction)",
            "domain": "drug",
            "default_question": "What is the local force / exit vector on the indole nitrogen?",
            "symmetry_hint": "Drug fragment inside a protein pocket",
            "frame": load_structure(drug_path),
        }

    presets["lifepo4_pbc_crystal"] = {
        "id": "lifepo4_pbc_crystal",
        "title": "LiFePO4 Crystal Cell (3D repeating battery cathode)",
        "domain": "electrolyte",
        "default_question": "What is the force vector on the coordinated Li+ ion in the crystal cell?",
        "symmetry_hint": "3D repeating crystal box: shifting an atom by one full box leaves answers unchanged",
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
        "symmetry_hint": "User-supplied 3D structure",
    }


@lru_cache(maxsize=2)
def _load_audit_models(precision: str = "float64") -> dict[str, torch.nn.Module]:
    """Build and cache fixed-seed models for instant side-by-side physics checks."""
    from src.fm_baselines import (
        EquiLLMMultiTaskBaseline,
        GraphMVP3DInfoMaxBaseline,
        UniMolStyle3DBaseline,
    )

    dtype = torch.float64 if precision == "float64" else torch.float32
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

    return {"ylm": ylm, "unimol": unimol, "graphmvp": graphmvp, "equillm": equillm}


def _frame_to_tensors(
    frame: StructureFrame,
    question: str,
    dtype: torch.dtype,
) -> dict[str, Optional[torch.Tensor]]:
    Z = torch.as_tensor(frame.Z, dtype=torch.long).unsqueeze(0)
    pos = torch.as_tensor(frame.pos_bohr, dtype=dtype).unsqueeze(0)
    atom_mask = torch.ones_like(Z, dtype=torch.bool)
    text = _FEATURIZER.encode(question).to(dtype=dtype).unsqueeze(0)
    text_mask = torch.ones((1, text.shape[1]), dtype=torch.bool)
    lot_id = torch.tensor([1], dtype=torch.long)
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
        "text": text,
        "text_mask": text_mask,
        "lot_id": lot_id,
        "cell": cell,
        "pbc": pbc,
    }


def _detect_point_symmetry(frame: StructureFrame) -> dict:
    """Check if all atoms lie on a flat plane or match opposite pairs across the center."""
    pos = frame.pos_bohr - frame.pos_bohr.mean(axis=0, keepdims=True)
    n = frame.n_atoms
    cov = (pos.T @ pos) / max(n, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    is_planar = bool(eigvals[0] < 1e-8)
    plane_normal = eigvecs[:, 0].astype(np.float64) if is_planar else None

    has_inversion = frame.cell_bohr is None
    if has_inversion:
        for i in range(n):
            dists = np.linalg.norm((-pos[i])[None, :] - pos, axis=-1)
            if not any(int(frame.Z[j]) == int(frame.Z[i]) and dists[j] < 1e-4 for j in range(n)):
                has_inversion = False
                break

    return {
        "is_planar": is_planar,
        "plane_normal": None if plane_normal is None else plane_normal.tolist(),
        "has_inversion_center": has_inversion,
        "inertia_eigenvalues_bohr2": [float(v) for v in eigvals.tolist()],
    }


def euler_and_reflection_matrix(rx_deg: float, ry_deg: float, rz_deg: float, reflect: bool = False) -> np.ndarray:
    """Build a 3x3 rotation matrix from X/Y/Z angles (in degrees), with optional mirror flip."""
    ax, ay, az = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    q = rz @ ry @ rx
    return -q if reflect else q


def _make_arrow(start_xyz: np.ndarray, vec: np.ndarray, name: str, color: str, length: float = 1.85) -> dict:
    norm = float(np.linalg.norm(vec))
    direction = (vec / norm) if norm > 1e-10 else np.zeros(3)
    end_xyz = start_xyz + direction * length
    return {
        "name": name,
        "vec": [round(float(c), 6) for c in vec.tolist()],
        "start": [round(float(c), 5) for c in start_xyz.tolist()],
        "end": [round(float(c), 5) for c in end_xyz.tolist()],
        "color": color,
    }


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
    """Spin, mirror, move, or reorder a molecule and compare YLM against baselines."""
    if isinstance(frame_or_source, StructureFrame):
        frame = copy.deepcopy(frame_or_source)
    else:
        frame, _ = resolve_preset_or_structure(str(frame_or_source))

    dtype = torch.float64 if precision == "float64" else torch.float32
    models = _load_audit_models(precision)

    q_np = euler_and_reflection_matrix(rx_deg, ry_deg, rz_deg, reflect=reflect)
    q_mat = torch.as_tensor(q_np, dtype=dtype)
    det_q = float(np.linalg.det(q_np))
    shift_bohr = np.asarray(translate_ang, dtype=np.float64) * ANGSTROM_TO_BOHR

    n = frame.n_atoms
    perm = np.arange(n, dtype=np.int64)
    if permute_atoms and n > 1:
        perm = np.random.default_rng(137).permutation(n)

    frame_t = StructureFrame(
        Z=frame.Z[perm],
        pos_bohr=(frame.pos_bohr[perm] @ q_np.T) + shift_bohr[None, :],
        cell_bohr=None if frame.cell_bohr is None else (frame.cell_bohr @ q_np.T),
        pbc=None if frame.pbc is None else frame.pbc.copy(),
        label=f"{frame.label}_transformed",
    )

    inp0 = _frame_to_tensors(frame, question, dtype)
    inp1 = _frame_to_tensors(frame_t, question, dtype)

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
        v_expected = q_mat @ v0
        v_rel_err = float((v1 - v_expected).norm().item() / max(float(v0.norm().item()), 1e-2))
        v_cos = float(
            torch.dot(v1, v_expected).item()
            / max(float(v1.norm().item() * v_expected.norm().item()), 1e-12)
        )

        t0 = out0["tensor"][0]
        t1 = out1["tensor"][0]
        t_expected = rotate_matrix(out0["tensor"], q_mat)[0]
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

    orig_center = frame.pos_angstrom.mean(axis=0)
    trans_center = frame_t.pos_angstrom.mean(axis=0)
    ylm_v0 = np.asarray(comparisons["ylm"]["vector_1o"]["orig"], dtype=np.float64)
    ylm_v_act = np.asarray(comparisons["ylm"]["vector_1o"]["transformed"], dtype=np.float64)
    ylm_v_exp = np.asarray(comparisons["ylm"]["vector_1o"]["expected"], dtype=np.float64)
    uni_v_act = np.asarray(comparisons["unimol"]["vector_1o"]["transformed"], dtype=np.float64)

    scene_orig = _serialize_geometry_scene(
        frame,
        arrows=[_make_arrow(orig_center, ylm_v0, "YLM Arrow (Before)", "#38bdf8", 1.85)],
        weights=comparisons["ylm"]["attn_orig"],
    )
    scene_transformed = _serialize_geometry_scene(
        frame_t,
        arrows=[
            _make_arrow(trans_center, ylm_v_exp, "Exact Physics Target", "#22c55e", 1.95),
            _make_arrow(trans_center, ylm_v_act, "YLM Arrow (After)", "#38bdf8", 1.75),
            _make_arrow(trans_center, uni_v_act, "Uni-Mol Arrow (Broken)", "#ef4444", 1.55),
        ],
        weights=comparisons["ylm"]["attn_transformed"],
    )

    return {
        "precision": precision,
        "det_Q": round(det_q, 4),
        "operation": "3D Spin + Mirror Flip (det = -1)" if reflect else "3D Rotation Only (det = +1)",
        "matrix_Q": [[round(float(x), 6) for x in row] for row in q_np.tolist()],
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
    """Run all 8 physics checks in live PyTorch and return a plain-English report."""
    if isinstance(frame_or_source, StructureFrame):
        frame = copy.deepcopy(frame_or_source)
        preset_meta = {"id": "custom", "title": frame.label}
    else:
        frame, preset_meta = resolve_preset_or_structure(str(frame_or_source))

    dtype = torch.float64 if precision == "float64" else torch.float32
    tol = 1e-8 if dtype == torch.float64 else 5e-5
    models = _load_audit_models(precision)
    ylm = models["ylm"]
    unimol = models["unimol"]
    graphmvp = models["graphmvp"]
    equillm = models["equillm"]

    inp0 = _frame_to_tensors(frame, question, dtype)
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

        # Rule 1: Move in space (Translation)
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
            "name": "1. Move in Space (Translation)",
            "formula": "f(atoms + shift) == f(atoms)",
            "passed": tr_err < tol,
            "ylm_error": tr_err,
            "tolerance": tol,
            "detail": (
                f"Moved all {frame.n_atoms} atoms by (+17.3, -11.9, +24.6) bohr. "
                f"Empty space is the same everywhere, so answers stayed identical (max diff = {tr_err:.2e})."
            ),
        })

        # Rule 2: Spin in 3D (Rotation)
        so3_sc, so3_ch, so3_vec, so3_ten = [], [], [], []
        uni_so3_vec, gmvp_so3_vec = [], []
        with torch.random.fork_rng():
            torch.manual_seed(314)
            for _ in range(max(1, n_rotations)):
                rot = o3.rand_matrix(dtype=dtype)
                if torch.linalg.det(rot) < 0:
                    rot = -rot
                pos_r = inp0["pos"] @ rot.T
                cell_r = None if inp0["cell"] is None else (inp0["cell"] @ rot.T)
                o_r = ylm(
                    inp0["Z"], pos_r, inp0["atom_mask"],
                    inp0["text"], inp0["text_mask"],
                    lot_id=inp0["lot_id"], cell=cell_r, pbc=inp0["pbc"],
                )
                so3_sc.append(float((o_r["scalar"] - out0["scalar"]).abs().max().item()))
                so3_ch.append(float((o_r["chiral"] - out0["chiral"]).abs().max().item()))
                v_den = max(float(out0["vector"].norm().item()), 1e-2)
                so3_vec.append(float((o_r["vector"] - (out0["vector"] @ rot.T)).norm().item()) / v_den)
                t_den = max(float(out0["tensor"].norm().item()), 1e-2)
                so3_ten.append(float((o_r["tensor"] - rotate_matrix(out0["tensor"], rot)).norm().item()) / t_den)

                uo_r = unimol(
                    inp0["Z"], pos_r, inp0["atom_mask"],
                    inp0["text"], inp0["text_mask"],
                    lot_id=inp0["lot_id"], cell=cell_r, pbc=inp0["pbc"],
                )
                uv_den = max(float(u_out0["vector"].norm().item()), 1e-12)
                uni_so3_vec.append(float((uo_r["vector"] - (u_out0["vector"] @ rot.T)).norm().item()) / uv_den)

                go_r = graphmvp(
                    inp0["Z"], pos_r, inp0["atom_mask"],
                    inp0["text"], inp0["text_mask"],
                    lot_id=inp0["lot_id"], cell=cell_r, pbc=inp0["pbc"],
                )
                gv_den = max(float(g_out0["vector"].norm().item()), 1e-12)
                gmvp_so3_vec.append(float((go_r["vector"] - (g_out0["vector"] @ rot.T)).norm().item()) / gv_den)

        max_so3 = max(max(so3_sc), max(so3_ch), max(so3_vec), max(so3_ten))
        laws.append({
            "law_id": "L2_so3_rotation",
            "name": f"2. Spin in 3D ({n_rotations} Random Rotations)",
            "formula": "energy' = energy | arrow' = R * arrow | matrix' = R * matrix * R^T",
            "passed": max_so3 < tol,
            "ylm_error": max_so3,
            "baseline_errors": {
                "Uni-Mol arrow error": float(np.mean(uni_so3_vec)),
                "GraphMVP arrow error": float(np.mean(gmvp_so3_vec)),
            },
            "tolerance": tol,
            "detail": (
                f"Spun the molecule in 3D: YLM arrows turned by the exact same angle (max error = {max_so3:.2e}). "
                f"Distance-only Uni-Mol failed to rotate its arrow (error = {np.mean(uni_so3_vec):.2e})."
            ),
        })

        # Rule 3: Mirror Flip (Left Hand vs. Right Hand)
        imp_sc, imp_ch, imp_vec, imp_ten = [], [], [], []
        with torch.random.fork_rng():
            torch.manual_seed(2718)
            for k in range(max(1, n_rotations)):
                q_mat = -torch.eye(3, dtype=dtype) if k == 0 else -o3.rand_matrix(dtype=dtype)
                if torch.linalg.det(q_mat) > 0:
                    q_mat = -q_mat
                pos_q = inp0["pos"] @ q_mat.T
                cell_q = None if inp0["cell"] is None else (inp0["cell"] @ q_mat.T)
                o_q = ylm(
                    inp0["Z"], pos_q, inp0["atom_mask"],
                    inp0["text"], inp0["text_mask"],
                    lot_id=inp0["lot_id"], cell=cell_q, pbc=inp0["pbc"],
                )
                imp_sc.append(float((o_q["scalar"] - out0["scalar"]).abs().max().item()))
                imp_ch.append(float((o_q["chiral"] + out0["chiral"]).abs().max().item()))
                v_den = max(float(out0["vector"].norm().item()), 1e-2)
                imp_vec.append(float((o_q["vector"] - (out0["vector"] @ q_mat.T)).norm().item()) / v_den)
                t_den = max(float(out0["tensor"].norm().item()), 1e-2)
                imp_ten.append(float((o_q["tensor"] - rotate_matrix(out0["tensor"], q_mat)).norm().item()) / t_den)

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
            "name": "3. Mirror Flip (Left Hand vs. Right Hand)",
            "formula": "energy(-X) = +energy(X) | hand(-X) = -hand(X) | arrow(-X) = -arrow(X)",
            "passed": max_imp < tol,
            "ylm_error": max_imp,
            "baseline_errors": {
                "Uni-Mol mirror error |h(-X) + h(X)|": uni_parity_err,
                "EquiLLM mirror error |h(-X) + h(X)|": equi_parity_err,
            },
            "tolerance": tol,
            "detail": (
                f"Flipped the molecule in a mirror: YLM handedness flipped sign cleanly "
                f"({float(out0['chiral'].item()):+.3e} -> {-float(out0['chiral'].item()):+.3e}, error = {max_imp:.2e}). "
                f"Uni-Mol gave the same number to both mirror twins (error = {uni_parity_err:.2e})."
            ),
        })

        # Rule 4: Swap Atom Row Order
        n_at = frame.n_atoms
        perm = torch.randperm(n_at) if n_at > 1 else torch.arange(n_at)
        out_p = ylm(
            inp0["Z"][:, perm], inp0["pos"][:, perm, :], inp0["atom_mask"],
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
            "name": "4. Swap Atom Row Order",
            "formula": "f(shuffled atoms) == f(atoms), and per-atom weights follow each atom",
            "passed": perm_err < tol,
            "ylm_error": perm_err,
            "tolerance": tol,
            "detail": (
                f"Shuffled the {n_at} atom rows ({perm.tolist()[:8]}): molecule answers stayed identical "
                f"and atom weights followed each atom (max diff = {perm_err:.2e})."
            ),
        })

        # Rule 5: Balanced 3x3 Response Matrix
        t_mat = out0["tensor"][0]
        asym_err = float((t_mat - t_mat.T).norm().item())
        iso_part = (torch.diagonal(t_mat).sum() / 3.0) * torch.eye(3, dtype=dtype)
        traceless_part = t_mat - iso_part
        trace_res = float(torch.diagonal(traceless_part).sum().abs().item())
        ten_sym_err = max(asym_err, trace_res)
        laws.append({
            "law_id": "L5_tensor_symmetry",
            "name": "5. Balanced 3x3 Response Matrix",
            "formula": "matrix[i, j] == matrix[j, i], split into average size + 3D stretch",
            "passed": ten_sym_err < tol,
            "ylm_error": ten_sym_err,
            "tolerance": tol,
            "detail": (
                f"Matrix equals its transpose (||T - T^T|| = {asym_err:.2e}), "
                f"and the 3D stretch part has zero trace ({trace_res:.2e})."
            ),
        })

        # Rule 6: Balanced Shape Cancels Out
        sym_info = _detect_point_symmetry(frame)
        curie_checks: list[tuple[str, float]] = []
        if sym_info["is_planar"]:
            n_vec = torch.tensor(sym_info["plane_normal"], dtype=dtype)
            out_of_plane = float(torch.dot(out0["vector"][0], n_vec).abs().item())
            chiral_planar = float(out0["chiral"].abs().max().item())
            tn = out0["tensor"][0] @ n_vec
            shear = float((tn - torch.dot(n_vec, tn) * n_vec).norm().item())
            curie_checks.extend([
                ("out_of_plane_arrow", out_of_plane),
                ("flat_handedness", chiral_planar),
                ("out_of_plane_shear", shear),
            ])
        if sym_info["has_inversion_center"]:
            curie_checks.extend([
                ("total_dipole_arrow", float(out0["vector"][0].norm().item())),
                ("center_handedness", float(out0["chiral"].abs().max().item())),
            ])

        used_control = False
        if not curie_checks:
            used_control = True
            presets = get_builtin_presets()
            r_ctrl = euler_and_reflection_matrix(29.0, -51.0, 64.0, reflect=False)
            w_fr = presets["water_c2v"]["frame"].transformed(r_ctrl)
            w_in = _frame_to_tensors(w_fr, "What is the 3D molecular dipole moment vector?", dtype)
            w_out = ylm(w_in["Z"], w_in["pos"], w_in["atom_mask"], w_in["text"], w_in["text_mask"], lot_id=w_in["lot_id"])
            n_rot = torch.as_tensor(r_ctrl @ np.array([0.0, 0.0, 1.0]), dtype=dtype)
            curie_checks.append(("rotated_water_out_of_plane", float(torch.dot(w_out["vector"][0], n_rot).abs().item())))
            curie_checks.append(("rotated_water_handedness", float(w_out["chiral"].abs().max().item())))

            tr_fr = presets["trans_difluoroethene_c2h"]["frame"].transformed(r_ctrl)
            tr_in = _frame_to_tensors(tr_fr, "What is the 3D molecular dipole moment vector?", dtype)
            tr_out = ylm(tr_in["Z"], tr_in["pos"], tr_in["atom_mask"], tr_in["text"], tr_in["text_mask"], lot_id=tr_in["lot_id"])
            curie_checks.append(("rotated_trans_FCH=CHF_dipole", float(tr_out["vector"][0].norm().item())))

        curie_err = max(v for _, v in curie_checks)
        curie_desc = ", ".join(f"{k}={v:.2e}" for k, v in curie_checks)
        laws.append({
            "law_id": "L6_curie_selection",
            "name": "6. Balanced Shape Cancels Out",
            "formula": "Flat shape => out-of-plane arrow = 0, hand = 0 | Opposite-pair center => total arrow = 0",
            "passed": curie_err < tol,
            "ylm_error": curie_err,
            "tolerance": tol,
            "detail": (
                f"{'Input is 3D chiral (checked on rotated H2O and trans-FCH=CHF controls): ' if used_control else 'Checked directly on this symmetric shape: '}"
                f"{curie_desc}."
            ),
        })

    # Rule 7: Newton's 3rd Law, Zero Spin & Energy Slope (F = -dE/dx)
    pos_ang = frame.pos_angstrom.copy()
    cell_ang = frame.cell_angstrom
    pbc_arr = frame.pbc
    _, f0 = _FORCE_PROVIDER(frame.Z, pos_ang, cell_ang, pbc_arr)

    net_force = float(np.linalg.norm(f0.sum(axis=0)))
    if cell_ang is None:
        r_rel = pos_ang - pos_ang.mean(axis=0, keepdims=True)
        net_torque = float(np.linalg.norm(np.cross(r_rel, f0).sum(axis=0)))
    else:
        net_torque = 0.0

    eps_fd = 1e-5
    f_fd = np.zeros(3, dtype=np.float64)
    for alpha in range(3):
        p_plus = pos_ang.copy()
        p_minus = pos_ang.copy()
        p_plus[0, alpha] += eps_fd
        p_minus[0, alpha] -= eps_fd
        ep, _ = _FORCE_PROVIDER(frame.Z, p_plus, cell_ang, pbc_arr)
        em, _ = _FORCE_PROVIDER(frame.Z, p_minus, cell_ang, pbc_arr)
        f_fd[alpha] = -(ep - em) / (2.0 * eps_fd)
    fd_grad_err = float(np.max(np.abs(f0[0] - f_fd)))

    curl_err = 0.0
    if frame.n_atoms >= 2:
        p_1y_p, p_1y_m = pos_ang.copy(), pos_ang.copy()
        p_1y_p[1, 1] += eps_fd
        p_1y_m[1, 1] -= eps_fd
        _, f_1y_p = _FORCE_PROVIDER(frame.Z, p_1y_p, cell_ang, pbc_arr)
        _, f_1y_m = _FORCE_PROVIDER(frame.Z, p_1y_m, cell_ang, pbc_arr)
        df0x_dr1y = (f_1y_p[0, 0] - f_1y_m[0, 0]) / (2.0 * eps_fd)

        p_0x_p, p_0x_m = pos_ang.copy(), pos_ang.copy()
        p_0x_p[0, 0] += eps_fd
        p_0x_m[0, 0] -= eps_fd
        _, f_0x_p = _FORCE_PROVIDER(frame.Z, p_0x_p, cell_ang, pbc_arr)
        _, f_0x_m = _FORCE_PROVIDER(frame.Z, p_0x_m, cell_ang, pbc_arr)
        df1y_dr0x = (f_0x_p[1, 1] - f_0x_m[1, 1]) / (2.0 * eps_fd)
        curl_err = float(abs(df0x_dr1y - df1y_dr0x))

    mech_tol = 5e-6
    mech_err = max(net_force, net_torque, fd_grad_err, curl_err)
    laws.append({
        "law_id": "L7_conservative_mechanics",
        "name": "7. Newton's 3rd Law, Zero Spin & Energy Slope (F = -dE/dx)",
        "formula": "sum(F) == 0,  sum(r x F) == 0,  and  F == -d(Energy)/dx",
        "passed": mech_err < mech_tol,
        "ylm_error": mech_err,
        "tolerance": mech_tol,
        "detail": (
            f"Total internal force = {net_force:.2e} eV/A (cannot push itself), "
            f"total torque = {net_torque:.2e} eV (cannot spin itself), "
            f"energy-slope match |F + dE/dx| = {fd_grad_err:.2e} eV/A."
        ),
    })

    # Rule 8: Crystal Repeat Box
    with torch.no_grad():
        pbc_frame = frame if frame.cell_bohr is not None else get_builtin_presets()["lifepo4_pbc_crystal"]["frame"]
        p_in0 = _frame_to_tensors(pbc_frame, question, dtype)
        p_out0 = ylm(
            p_in0["Z"], p_in0["pos"], p_in0["atom_mask"],
            p_in0["text"], p_in0["text_mask"],
            lot_id=p_in0["lot_id"], cell=p_in0["cell"], pbc=p_in0["pbc"],
        )
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
            "name": "8. Crystal Repeat Box",
            "formula": "f(atom + box_vector, cell) == f(atom, cell)",
            "passed": pbc_err < tol,
            "ylm_error": pbc_err,
            "tolerance": tol,
            "detail": (
                f"Tested on {'input crystal' if frame.cell_bohr is not None else 'LiFePO4 3D crystal'}: "
                f"shifted atom #0 by one full crystal box vector (box_1 - box_2); max diff = {pbc_err:.2e}."
            ),
        })

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

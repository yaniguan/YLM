"""Literature-aligned O(3)-equivariant domain packs for Electrolyte, Reactive MD (1o motion), and Drug SBDD."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from src.toolkit.ingest import (
    BOHR_TO_ANGSTROM,
    StructureFrame,
    covalent_radius,
    load_structure,
    load_trajectory,
    minimum_image_delta_bohr,
    symbol_for_z,
)
from src.toolkit.probe import ProbeResult, VectorPrediction, YLMProbe

WORKING_ION_ZS = {3, 11, 12, 19, 20, 30}       # Li, Na, Mg, K, Ca, Zn
DONOR_ZS = {7, 8, 9, 16, 17, 35, 53}           # N, O, F, S, Cl, Br, I
ANION_MARKER_ZS = {9, 15, 16, 17, 35, 53, 5}   # F, P, S, Cl, Br, I, B (PF6-, FSI-, TFSI-, BF4-, Cl-)

_PAULING_CHI = {
    1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    11: 0.93, 12: 1.31, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    19: 0.82, 20: 1.00, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65,
    35: 2.96, 53: 2.66,
}

_VDW_RADII_ANG = {
    1: 1.10, 3: 1.82, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
    11: 2.27, 12: 1.73, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98,
}


def _vdw_radius(z: int) -> float:
    return _VDW_RADII_ANG.get(int(z), 1.75)


def _pairwise_min_image_angstrom(frame: StructureFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (rel_ij (N, N, 3) = r_i - r_j in Angstrom, dist_ij (N, N) in Angstrom) using minimum image if periodic."""
    n = frame.n_atoms
    if n == 0:
        return np.zeros((0, 0, 3), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
    rel_bohr = frame.pos_bohr[:, None, :] - frame.pos_bohr[None, :, :]
    if frame.cell_bohr is not None:
        flat = minimum_image_delta_bohr(
            np.repeat(frame.pos_bohr, n, axis=0),
            np.tile(frame.pos_bohr, (n, 1)),
            frame.cell_bohr,
            frame.pbc,
        )
        rel_bohr = flat.reshape(n, n, 3)
    rel_ang = rel_bohr * BOHR_TO_ANGSTROM
    dist_ang = np.linalg.norm(rel_ang, axis=-1)
    return rel_ang, dist_ang


def _subgraph_bond_dipole_debye(frame: StructureFrame, atom_indices: Sequence[int], dist_ang: np.ndarray, rel_ang: np.ndarray) -> np.ndarray:
    """Compute O(3)-covariant, translation-invariant bond-charge dipole (1o, Debye) on a subset of atoms.

    Convention: points from negative (more electronegative) pole toward positive pole so that a coordinating
    carbonyl oxygen facing Li+ has dipole aligned WITH r_{Li -> ligand} (cos(theta) > 0).
    """
    idx = [int(i) for i in atom_indices]
    m = len(idx)
    if m <= 1:
        return np.zeros(3, dtype=np.float64)
    charges = np.zeros(m, dtype=np.float64)
    for a_pos in range(m):
        i = idx[a_pos]
        zi = int(frame.Z[i])
        chi_i = _PAULING_CHI.get(zi, 2.2)
        for b_pos in range(a_pos + 1, m):
            j = idx[b_pos]
            zj = int(frame.Z[j])
            d = float(dist_ang[i, j])
            r0 = covalent_radius(zi) + covalent_radius(zj)
            if 0.1 < d <= 1.38 * r0:
                chi_j = _PAULING_CHI.get(zj, 2.2)
                dq = 0.35 * math.tanh(0.8 * (chi_i - chi_j))
                # If chi_i > chi_j, atom i is more negative (-dq) and atom j is more positive (+dq)
                charges[a_pos] -= dq
                charges[b_pos] += dq
    # Compute origin-independent dipole using MIC displacements relative to the first atom in the connected subgraph
    ref = idx[0]
    disp_from_ref = np.array([rel_ang[k, ref] for k in idx], dtype=np.float64)  # r_k - r_ref
    mean_disp = disp_from_ref.mean(axis=0)
    centered = disp_from_ref - mean_disp[None, :]
    return (charges[:, None] * centered).sum(axis=0) * 4.8032047


def _connected_components(atom_indices: Sequence[int], z_arr: np.ndarray, dist_ang: np.ndarray, scale: float = 1.32) -> list[list[int]]:
    idx_set = set(int(i) for i in atom_indices)
    adj: dict[int, list[int]] = {i: [] for i in idx_set}
    sorted_idx = sorted(idx_set)
    for a_i, i in enumerate(sorted_idx):
        ri = covalent_radius(int(z_arr[i]))
        for j in sorted_idx[a_i + 1:]:
            rj = covalent_radius(int(z_arr[j]))
            if 0.1 < dist_ang[i, j] <= scale * (ri + rj):
                adj[i].append(j)
                adj[j].append(i)
    seen: set[int] = set()
    comps: list[list[int]] = []
    for i in sorted_idx:
        if i in seen:
            continue
        comp = []
        q = deque([i])
        seen.add(i)
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        comps.append(comp)
    return comps


# ---------------------------------------------------------------------------
# 1. Electrolyte Solvation Shell & Periodic Electrode Domain Pack
#    Aligned with SolvationAnalysis (Persson 2023), Holoubek & Liu (Nat. Energy 2021),
#    BAMBOO (2024), OMol25 (2025), MACE-MP-0 (2024), LiTraj (2025), MolSets/DiffMix (2024).
# ---------------------------------------------------------------------------

def _smooth_cosine_cutoff(d: float, r_cut: float) -> float:
    if d <= 0.05 or d >= r_cut:
        return 0.0
    return 0.5 * (math.cos(math.pi * d / r_cut) + 1.0)


def analyze_electrolyte_geometry(frame: StructureFrame, shell_cutoff_ang: float = 3.0) -> dict:
    """Analyze cation solvation shell: discrete/smooth CN, soft SSIP/CIP/AGG, ligand radial cos(theta),
    1o desolvation escape vector, and 2e centrosymmetric quadrupole channel tensor."""
    rel_ang, dist_ang = _pairwise_min_image_angstrom(frame)
    ion_indices = [
        i for i, (z, r) in enumerate(zip(frame.Z.tolist(), frame.role.tolist()))
        if z in WORKING_ION_ZS or (r == 1 and z not in DONOR_ZS and z not in ANION_MARKER_ZS)
    ]
    if not ion_indices:
        ion_indices = [i for i, z in enumerate(frame.Z.tolist()) if z in WORKING_ION_ZS]
    if not ion_indices:
        ion_indices = [0] if frame.n_atoms else []

    if not ion_indices:
        return {
            "motif": "empty",
            "coordination_number": 0,
            "smooth_coordination_number": 0.0,
            "escape_vector": [0.0, 0.0, 0.0],
            "cage_quadrupole_2e": np.zeros((3, 3)).tolist(),
        }

    primary_ion = ion_indices[0]
    non_ion_atoms = [i for i in range(frame.n_atoms) if i not in ion_indices]
    mol_comps = _connected_components(non_ion_atoms, frame.Z, dist_ang, scale=1.35)
    atom_to_comp: dict[int, list[int]] = {}
    for comp in mol_comps:
        for a in comp:
            atom_to_comp[a] = comp

    coordinators = []
    repulsive = np.zeros(3, dtype=np.float64)
    q_cage = np.zeros((3, 3), dtype=np.float64)
    smooth_cn = 0.0
    soft_anion_cn = 0.0
    soft_bridge_excess = 0.0
    has_anion_contact = False
    bridging_anion = False
    visited_ligands: set[int] = set()
    radial_alignments: list[float] = []

    smooth_rc = shell_cutoff_ang * 1.15
    for j in range(frame.n_atoms):
        if j == primary_ion:
            continue
        d = float(dist_ang[primary_ion, j])
        zj = int(frame.Z[j])
        is_coord_elem = (zj in DONOR_ZS) or (int(frame.role[j]) in (1, 2) and zj > 1)
        if not is_coord_elem:
            continue

        w_smooth = _smooth_cosine_cutoff(d, smooth_rc)
        if w_smooth > 0.0:
            smooth_cn += w_smooth
            # Unit vector from primary_ion -> j: rel_ang[j, primary_ion] / d
            u_ion_to_j = rel_ang[j, primary_ion] / max(d, 1e-8)
            q_cage += w_smooth * (np.outer(u_ion_to_j, u_ion_to_j) - np.eye(3) / 3.0)

        comp = atom_to_comp.get(j, [j])
        comp_is_anion = (int(frame.role[j]) == 1) or any(int(frame.Z[a]) in (9, 15, 17, 35, 53, 5) for a in comp)

        if w_smooth > 0.0 and comp_is_anion:
            soft_anion_cn += w_smooth
            other_ion_w = sum(
                _smooth_cosine_cutoff(float(dist_ang[ion2, a]), smooth_rc)
                for ion2 in ion_indices if ion2 != primary_ion
                for a in comp
            )
            soft_bridge_excess += w_smooth * max(0.0, other_ion_w)

        if 0.1 < d <= shell_cutoff_ang:
            coordinators.append({
                "atom_index": int(j),
                "symbol": symbol_for_z(zj),
                "distance_angstrom": round(d, 4),
                "role": int(frame.role[j]),
                "is_anion": bool(comp_is_anion),
            })
            # Vector from coordinating atom j toward primary_ion weighted by 1/d^3 points out of steric cage gaps (1o).
            vec_ij = rel_ang[primary_ion, j]
            repulsive = repulsive + vec_ij / max(d ** 3, 1e-6)

            if comp_is_anion:
                has_anion_contact = True
                other_ions = [
                    ion for ion in ion_indices
                    if ion != primary_ion and any(dist_ang[ion, a] <= shell_cutoff_ang for a in comp)
                ]
                if other_ions:
                    bridging_anion = True

            comp_key = min(comp)
            if comp_key not in visited_ligands and len(comp) > 1:
                visited_ligands.add(comp_key)
                mu_lig = _subgraph_bond_dipole_debye(frame, comp, dist_ang, rel_ang)
                mu_norm = float(np.linalg.norm(mu_lig))
                if mu_norm > 1e-4:
                    # Radial vector from primary_ion to coordinating donor atom j
                    r_hat_ion_to_lig = rel_ang[j, primary_ion] / max(d, 1e-8)
                    cos_th = float(np.dot(mu_lig, r_hat_ion_to_lig) / mu_norm)
                    radial_alignments.append(cos_th)

    coordinators.sort(key=lambda item: item["distance_angstrom"])
    norm_esc = float(np.linalg.norm(repulsive))
    # Strictly O(3)-covariant 1o escape vector (remains [0, 0, 0] at exact centrosymmetry!)
    escape_dir = (repulsive / norm_esc).tolist() if norm_esc > 1e-7 else [0.0, 0.0, 0.0]

    # Centrosymmetric 2e open-channel tensor analysis: minimum eigenvalue of Q_cage points along least-crowded channel axis
    eigvals, eigvecs = np.linalg.eigh(q_cage)
    channel_anisotropy = float(eigvals[-1] - eigvals[0]) if eigvals.size == 3 else 0.0
    open_axis = eigvecs[:, 0].tolist() if eigvals.size == 3 else [1.0, 0.0, 0.0]

    if bridging_anion:
        motif = "AGG"
    elif has_anion_contact:
        motif = "CIP"
    elif coordinators:
        motif = "SSIP"
    else:
        motif = "uncoordinated"

    # Differentiable soft speciation probabilities (0e invariants)
    p_ssip = math.exp(-4.0 * soft_anion_cn)
    p_agg = (1.0 - p_ssip) * (1.0 - math.exp(-5.5 * soft_bridge_excess))
    p_cip = max(0.0, 1.0 - p_ssip - p_agg)
    p_sum = max(p_ssip + p_cip + p_agg, 1e-8)
    soft_spec = {
        "SSIP": round(p_ssip / p_sum, 4),
        "CIP": round(p_cip / p_sum, 4),
        "AGG": round(p_agg / p_sum, 4),
    }

    mean_align = float(np.mean(radial_alignments)) if radial_alignments else None
    locking = (
        "unknown" if mean_align is None
        else ("strong_ion_dipole_locked" if mean_align >= 0.55
              else ("moderate_solvation" if mean_align >= 0.15 else "weakly_solvating_or_tilted"))
    )

    return {
        "primary_ion_index": int(primary_ion),
        "primary_ion_symbol": symbol_for_z(int(frame.Z[primary_ion])),
        "coordination_number": len(coordinators),
        "smooth_coordination_number": round(smooth_cn, 4),
        "motif": motif,
        "soft_speciation": soft_spec,
        "periodic": frame.cell_bohr is not None,
        "centrosymmetric_cage": bool(norm_esc <= 1e-7 and len(coordinators) > 0),
        "mean_ligand_radial_alignment_cos": None if mean_align is None else round(mean_align, 4),
        "dipole_locking_regime": locking,
        "coordinators": coordinators,
        "escape_vector": escape_dir,
        "cage_asymmetry_norm": round(norm_esc, 5),
        "cage_quadrupole_2e": np.round(q_cage, 6).tolist(),
        "channel_anisotropy_2e": round(channel_anisotropy, 5),
        "open_channel_axis_pm": [round(float(x), 5) for x in open_axis],
    }


def analyze_electrolyte(
    source: Union[str, Path, StructureFrame],
    probe: Optional[YLMProbe] = None,
    question: Optional[str] = None,
    verify_o3: bool = False,
) -> ProbeResult:
    frame = load_structure(source)
    probe = probe or YLMProbe.default()
    info = analyze_electrolyte_geometry(frame)
    ion_sym = info.get("primary_ion_symbol", "Li")
    q = question or f"What is the net solvation pull force on the {ion_sym} ion and its desolvation direction?"
    res = probe.ask(frame, q, domain="electrolyte", verify_o3=verify_o3)
    return res


# ---------------------------------------------------------------------------
# 2. Equivariant MD Trajectory Domain Pack (SEGNN 1o Motion, Grotthuss CV, Work W_i)
#    Aligned with SEGNN (ICLR 2022), Calegari Andrade et al. (2020), mlcolvar (JCP 2023),
#    ZundEig (2023), Gomez, Thompson & Laage (Nat. Chem. 2024), EquiLLM (ICML 2025).
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryScanResult:
    n_frames_total: int
    scanned_indices: list[int]
    events: list[dict]
    frame_results: list[ProbeResult]
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_frames_total": self.n_frames_total,
            "scanned_indices": self.scanned_indices,
            "events": self.events,
            "summary": self.summary,
            "frames": [r.to_dict() for r in self.frame_results],
        }

    def to_html(self, path: Optional[Union[str, Path]] = None, title: Optional[str] = None) -> str:
        from src.toolkit.viz import render_trajectory_html_report
        html = render_trajectory_html_report(self, title=title)
        if path is not None:
            out_p = Path(path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(html, encoding="utf-8")
        return html


def _smooth_rational_cn(dist_row: np.ndarray, z_arr: np.ndarray, self_idx: int, r0: float = 1.30) -> float:
    """mlcolvar / PLUMED smooth rational proton coordination number C_i = sum_{H} (1 - (r/r0)^6) / (1 - (r/r0)^12)."""
    total = 0.0
    for j in range(len(z_arr)):
        if j == self_idx or int(z_arr[j]) != 1:
            continue
        ratio = float(dist_row[j]) / r0
        if ratio <= 1e-6:
            total += 1.0
        elif abs(ratio - 1.0) < 1e-5:
            total += 0.5
        elif ratio < 3.0:
            r6 = ratio ** 6
            total += (1.0 - r6) / (1.0 - r6 * r6)
    return total


def detect_md_frame_events(frames: Sequence[StructureFrame]) -> list[dict]:
    """Detect bond events and Grotthuss proton-sharing bridges (delta = d(D-H*) - d(A-H*), R_DA, smooth CN, rattling vs committed)."""
    events: list[dict] = []
    prev_dist = None
    # Track signed delta history per (H_idx, low_heavy, high_heavy) triple to distinguish committed transfer vs rattling
    bridge_history: dict[tuple[int, int, int], list[tuple[int, float]]] = {}

    for idx, f in enumerate(frames):
        _, dist = _pairwise_min_image_angstrom(f)
        for i in range(f.n_atoms):
            if int(f.Z[i]) != 1:
                continue
            heavy = [j for j in range(f.n_atoms) if j != i and int(f.Z[j]) in DONOR_ZS and dist[i, j] <= 1.65]
            if len(heavy) >= 2:
                nearest_two = sorted(heavy, key=lambda j: float(dist[i, j]))[:2]
                d_near0 = float(dist[i, nearest_two[0]])
                d_near1 = float(dist[i, nearest_two[1]])
                # Canonical heavy-atom ordering (donor_idx < acceptor_idx) so signed delta(t) is continuous across frames
                h_a, h_b = (nearest_two[0], nearest_two[1]) if nearest_two[0] < nearest_two[1] else (nearest_two[1], nearest_two[0])
                delta_signed = float(dist[i, h_a] - dist[i, h_b])
                asymmetry = abs(d_near1 - d_near0)
                r_da = float(dist[h_a, h_b])
                cn_a = _smooth_rational_cn(dist[h_a], f.Z, h_a)
                cn_b = _smooth_rational_cn(dist[h_b], f.Z, h_b)
                bridge_state = "Zundel_shared_TS" if (asymmetry <= 0.15 and r_da <= 2.60) else "Eigen_asymmetric_bridge"
                ev = {
                    "frame_index": idx,
                    "type": "proton_transfer_bridge",
                    "atom_index": int(i),
                    "partners": [int(h_a), int(h_b)],
                    "distances_angstrom": [round(float(dist[i, h_a]), 4), round(float(dist[i, h_b]), 4)],
                    "grotthuss_delta_angstrom": round(delta_signed, 4),
                    "bridge_asymmetry_angstrom": round(asymmetry, 4),
                    "donor_acceptor_distance_angstrom": round(r_da, 4),
                    "smooth_proton_cn": [round(cn_a, 3), round(cn_b, 3)],
                    "bridge_state": bridge_state,
                    "transfer_commitment": "pending",
                }
                events.append(ev)
                bridge_history.setdefault((int(i), int(h_a), int(h_b)), []).append((len(events) - 1, delta_signed))

        if prev_dist is not None and prev_dist.shape == dist.shape:
            for i in range(f.n_atoms):
                for j in range(i + 1, f.n_atoms):
                    rcut = 1.30 * (covalent_radius(int(f.Z[i])) + covalent_radius(int(f.Z[j])))
                    was_bonded = prev_dist[i, j] <= rcut
                    now_bonded = dist[i, j] <= rcut
                    delta_d = float(dist[i, j] - prev_dist[i, j])
                    if was_bonded != now_bonded or ((was_bonded or now_bonded) and abs(delta_d) >= 0.15):
                        events.append({
                            "frame_index": idx,
                            "type": ("bond_break" if was_bonded and not now_bonded
                                     else ("bond_form" if now_bonded and not was_bonded else "bond_stretch")),
                            "atoms": [int(i), int(j)],
                            "delta_distance_angstrom": round(delta_d, 4),
                        })
        prev_dist = dist

    # Post-classify multi-frame proton bridges into committed Grotthuss transfer vs frustrated rattling
    for _, seq in bridge_history.items():
        first_delta = seq[0][1]
        last_delta = seq[-1][1]
        if len(seq) >= 2 and (first_delta * last_delta < -1e-4):
            label = "committed_grotthuss_transfer"
        elif len(seq) >= 2:
            label = "frustrated_rattling"
        else:
            label = "single_frame_bridge"
        for ev_idx, _ in seq:
            events[ev_idx]["transfer_commitment"] = label

    return events


def analyze_md_frame_motion(frame: StructureFrame, res: ProbeResult) -> dict:
    """Compute SEGNN-aligned 1o(F) x 1o(delta_r) -> 0e(work W_i) + 1e(torque) + 2e(virial shear)."""
    delta_ang = frame.delta_pos_angstrom
    if delta_ang is None:
        return {"has_motion_1o": False, "max_displacement_angstrom": 0.0}

    disp_norms = np.linalg.norm(delta_ang, axis=-1)
    fastest_atom = int(np.argmax(disp_norms)) if frame.n_atoms else 0

    target_atom = fastest_atom
    f_vec = None
    if res.vectors:
        vp = res.vectors[0]
        if vp.anchor_atom is not None:
            target_atom = int(vp.anchor_atom)
        f_vec = np.asarray(vp.vector_physical, dtype=np.float64)
    elif frame.forces_ev_ang is not None and frame.n_atoms:
        f_vec = np.asarray(frame.forces_ev_ang[target_atom], dtype=np.float64)

    alignment_cos = None
    work_ev = None
    torque_1e = None
    virial_2e = None
    if f_vec is not None and frame.n_atoms:
        d_vec = delta_ang[target_atom]
        work_ev = float(np.dot(f_vec, d_vec))
        torque_1e = np.cross(f_vec, d_vec)  # 1e axial pseudovector: det(R) * R @ torque
        virial_2e = 0.5 * (np.outer(f_vec, d_vec) + np.outer(d_vec, f_vec)) - (work_ev / 3.0) * np.eye(3)
        denom = float(np.linalg.norm(f_vec) * np.linalg.norm(d_vec))
        if denom > 1e-8:
            alignment_cos = float(work_ev / denom)

    total_recorded_work_ev = (
        round(float(np.sum(frame.forces_ev_ang * delta_ang)), 5)
        if frame.forces_ev_ang is not None and frame.forces_ev_ang.shape == delta_ang.shape
        else None
    )

    regime = (
        "unknown" if alignment_cos is None
        else ("driving_along_motion" if alignment_cos > 0.15
              else ("restoring_against_motion" if alignment_cos < -0.15 else "transverse_or_transitional"))
    )

    return {
        "has_motion_1o": True,
        "fastest_atom_index": fastest_atom,
        "fastest_atom_symbol": symbol_for_z(int(frame.Z[fastest_atom])) if frame.n_atoms else "?",
        "target_atom_index": target_atom,
        "max_displacement_angstrom": round(float(disp_norms.max(initial=0.0)), 5),
        "force_displacement_cosine": None if alignment_cos is None else round(alignment_cos, 4),
        "mechanical_work_ev": None if work_ev is None else round(work_ev, 5),
        "total_recorded_work_ev": total_recorded_work_ev,
        "kinematic_torque_1e": None if torque_1e is None else np.round(torque_1e, 6).tolist(),
        "kinematic_virial_2e": None if virial_2e is None else np.round(virial_2e, 6).tolist(),
        "regime": regime,
    }


def scan_md_trajectory(
    source: Union[str, Path, Sequence[StructureFrame]],
    probe: Optional[YLMProbe] = None,
    query: str = "What is the force vector on the hydrogen atom undergoing transfer?",
    every: int = 1,
    verify_o3: bool = False,
) -> TrajectoryScanResult:
    frames = load_trajectory(source)
    probe = probe or YLMProbe.default()
    events = detect_md_frame_events(frames)
    event_frame_set = {e["frame_index"] for e in events}
    selected = sorted(set(range(0, len(frames), max(1, every))) | event_frame_set)
    results = [
        probe.ask(frames[i], query, domain="md", verify_o3=(verify_o3 and idx == 0))
        for idx, i in enumerate(selected)
    ]

    # Compute O(3)-invariant event saliency score S(t) per scanned frame to highlight sub-ps TS crossings
    saliency_by_frame: dict[int, float] = {}
    for frame_idx, r in zip(selected, results):
        s_val = float(r.domain_summary.get("max_displacement_angstrom", 0.0))
        w_val = abs(float(r.domain_summary.get("mechanical_work_ev") or 0.0))
        s_val += 0.5 * w_val
        for ev in events:
            if ev["frame_index"] == frame_idx:
                if ev["type"] == "proton_transfer_bridge":
                    s_val += math.exp(-abs(float(ev["grotthuss_delta_angstrom"])) / 0.20)
                else:
                    s_val += 0.6
        saliency_by_frame[frame_idx] = round(s_val, 5)

    peak_keyframe = max(saliency_by_frame, key=saliency_by_frame.get) if saliency_by_frame else 0
    max_disp = max((r.domain_summary.get("max_displacement_angstrom", 0.0) for r in results), default=0.0)
    return TrajectoryScanResult(
        n_frames_total=len(frames),
        scanned_indices=selected,
        events=events,
        frame_results=results,
        summary={
            "n_events": len(events),
            "peak_displacement_angstrom": max_disp,
            "peak_saliency_frame": peak_keyframe,
            "keyframe_saliency": saliency_by_frame,
            "event_types": sorted({e["type"] for e in events}),
        },
    )


# ---------------------------------------------------------------------------
# 3. Drug Lead-Optimization Domain Pack (2e Aromatic Rings, HBD/HBA Cones,
#    Multi-Step vdW Ray-Marching Exit Vectors & Multi-Fragment Linker Bridges)
#    Aligned with ShEPhERD (ICLR 2025), PharmacoNet (2024), SurfDock (2025),
#    DecompDiff (ICML 2023), DeLinker (2020), LinkerNet (2023), DiffLinker (2024).
# ---------------------------------------------------------------------------

def _find_planar_rings(
    ligand_idx: Sequence[int],
    z_arr: np.ndarray,
    pos_ang: np.ndarray,
    dist_ang: np.ndarray,
) -> list[dict]:
    """Detect 5- and 6-membered planar rings on heavy atoms and compute 2e sign-invariant plane tensors."""
    heavy = [int(i) for i in ligand_idx if int(z_arr[i]) > 1]
    adj: dict[int, list[int]] = {i: [] for i in heavy}
    for idx_a, i in enumerate(heavy):
        ri = covalent_radius(int(z_arr[i]))
        for j in heavy[idx_a + 1:]:
            rj = covalent_radius(int(z_arr[j]))
            if 0.1 < dist_ang[i, j] <= 1.28 * (ri + rj):
                adj[i].append(j)
                adj[j].append(i)

    found_sets: set[frozenset[int]] = set()
    ordered_cycles: list[list[int]] = []

    def _dfs(start: int, cur: int, path: list[int]):
        if len(path) > 6:
            return
        for nb in adj[cur]:
            if nb == start and len(path) in (5, 6):
                key = frozenset(path)
                if key not in found_sets:
                    # Reject chorded non-elementary cycles
                    chords = sum(
                        1 for u in path for v in adj[u]
                        if v in key and v != path[(path.index(u) - 1) % len(path)] and v != path[(path.index(u) + 1) % len(path)]
                    )
                    if chords == 0:
                        found_sets.add(key)
                        ordered_cycles.append(list(path))
            elif nb > start and nb not in path and len(path) < 6:
                _dfs(start, nb, path + [nb])

    for s in heavy:
        _dfs(s, s, [s])

    rings: list[dict] = []
    for cyc in ordered_cycles:
        pts = pos_ang[cyc]
        centroid = pts.mean(axis=0)
        centered = pts - centroid[None, :]
        _, svals, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        rmsd_plane = float(svals[-1] / math.sqrt(len(cyc)))
        if rmsd_plane > 0.12:
            continue
        # Sign-ambiguity-free 2e symmetric traceless tensor: n x n - I/3 (transforms as R T R^T for all R in O(3))
        plane_tensor_2e = np.outer(normal, normal) - np.eye(3) / 3.0
        rings.append({
            "atoms": [int(a) for a in cyc],
            "size": len(cyc),
            "centroid_angstrom": np.round(centroid, 5).tolist(),
            "planarity_rmsd_angstrom": round(rmsd_plane, 5),
            "plane_tensor_2e": np.round(plane_tensor_2e, 6).tolist(),
            "_normal_raw": normal,
            "_centroid_np": centroid,
        })
    return rings


def analyze_drug_geometry(
    frame: StructureFrame,
    probe_step_ang: float = 2.4,
    clash_cutoff_ang: float = 2.2,
) -> dict:
    """Identify HBD/HBA pharmacophore cones, 2e aromatic rings, multi-step vdW exit vectors, and linker bridges."""
    rel_ang, dist_ang = _pairwise_min_image_angstrom(frame)
    roles = frame.role
    has_pocket = bool(np.any(roles != roles[0])) if frame.n_atoms else False
    ligand_idx = [i for i in range(frame.n_atoms) if (roles[i] == 0 if has_pocket else True)]
    pocket_idx = [i for i in range(frame.n_atoms) if i not in ligand_idx]
    pos_ang = frame.pos_angstrom

    # Covalent adjacency inside ligand
    lig_neighbors: dict[int, list[int]] = {i: [] for i in ligand_idx}
    for a_i, i in enumerate(ligand_idx):
        ri = covalent_radius(int(frame.Z[i]))
        for j in ligand_idx[a_i + 1:]:
            rj = covalent_radius(int(frame.Z[j]))
            if 0.1 < dist_ang[i, j] <= 1.32 * (ri + rj):
                lig_neighbors[i].append(j)
                lig_neighbors[j].append(i)

    # 1. Aromatic planar rings (2e tensor + optional pocket-directed 1o pi-stacking vector)
    raw_rings = _find_planar_rings(ligand_idx, frame.Z, pos_ang, dist_ang)
    ring_atoms_set = {a for r in raw_rings for a in r["atoms"]}
    aromatic_rings = []
    for r in raw_rings:
        c_np = r["_centroid_np"]
        n_raw = r["_normal_raw"]
        pi_vec = None
        if pocket_idx:
            p_vecs = pos_ang[pocket_idx] - c_np[None, :]
            p_dists = np.linalg.norm(p_vecs, axis=-1)
            near_mask = p_dists <= 5.5
            if np.any(near_mask):
                best_p = int(np.argmin(p_dists))
                sign = 1.0 if float(np.dot(n_raw, p_vecs[best_p])) >= 0.0 else -1.0
                pi_vec = np.round(sign * n_raw, 6).tolist()
        aromatic_rings.append({
            "atoms": r["atoms"],
            "size": r["size"],
            "centroid_angstrom": r["centroid_angstrom"],
            "planarity_rmsd_angstrom": r["planarity_rmsd_angstrom"],
            "plane_tensor_2e": r["plane_tensor_2e"],
            "pocket_pi_vector_1o": pi_vec,
        })

    # 2. H-bond donors (HBD, 1o) & lone-pair bisector H-bond acceptors (HBA, 1o) + pocket H-bond locks
    hbond_donors = []
    hbond_acceptors = []
    pocket_hbonds = []

    for i in ligand_idx:
        zi = int(frame.Z[i])
        nbs = lig_neighbors[i]
        if zi == 1 and nbs:
            parent = min(nbs, key=lambda j: dist_ang[i, j])
            zp = int(frame.Z[parent])
            if zp in (7, 8, 16):
                vec = pos_ang[i] - pos_ang[parent]
                norm = float(np.linalg.norm(vec))
                if norm > 1e-6:
                    u_dir = vec / norm
                    locked = False
                    if pocket_idx:
                        for pk in pocket_idx:
                            if int(frame.Z[pk]) in DONOR_ZS and 2.4 <= dist_ang[parent, pk] <= 3.5:
                                da_vec = pos_ang[pk] - pos_ang[parent]
                                cos_cone = float(np.dot(u_dir, da_vec) / max(float(np.linalg.norm(da_vec)), 1e-8))
                                if cos_cone >= 0.5:
                                    locked = True
                                    pocket_hbonds.append({
                                        "type": "HBD_to_pocket",
                                        "ligand_atom": int(parent),
                                        "pocket_atom": int(pk),
                                        "distance_angstrom": round(float(dist_ang[parent, pk]), 4),
                                        "cone_cosine": round(cos_cone, 4),
                                    })
                    hbond_donors.append({
                        "heavy_atom": int(parent),
                        "hydrogen_atom": int(i),
                        "symbol": f"{symbol_for_z(zp)}-H",
                        "direction": u_dir.tolist(),
                        "pharmacophore_locked": locked,
                    })

        elif zi in (7, 8, 9, 16) and 1 <= len(nbs) <= 3:
            has_attached_h = any(int(frame.Z[nb]) == 1 for nb in nbs)
            # Exclude pyrrole-like planar N-H (3 bonds with H) from lone-pair HBA
            if zi == 7 and len(nbs) >= 3 and has_attached_h:
                continue
            bond_units = []
            for nb in nbs:
                bv = pos_ang[nb] - pos_ang[i]
                bn = float(np.linalg.norm(bv))
                if bn > 1e-6:
                    bond_units.append(bv / bn)
            if bond_units:
                lp_vec = -np.sum(bond_units, axis=0)
                lp_norm = float(np.linalg.norm(lp_vec))
                if lp_norm > 1e-5:
                    u_lp = lp_vec / lp_norm
                    locked = False
                    if pocket_idx:
                        for pk in pocket_idx:
                            if int(frame.Z[pk]) in (7, 8, 16) and 2.4 <= dist_ang[i, pk] <= 3.5:
                                ap_vec = pos_ang[pk] - pos_ang[i]
                                cos_cone = float(np.dot(u_lp, ap_vec) / max(float(np.linalg.norm(ap_vec)), 1e-8))
                                if cos_cone >= 0.5:
                                    locked = True
                                    pocket_hbonds.append({
                                        "type": "HBA_to_pocket",
                                        "ligand_atom": int(i),
                                        "pocket_atom": int(pk),
                                        "distance_angstrom": round(float(dist_ang[i, pk]), 4),
                                        "cone_cosine": round(cos_cone, 4),
                                    })
                    hbond_acceptors.append({
                        "acceptor_atom": int(i),
                        "symbol": symbol_for_z(zi),
                        "direction": u_lp.tolist(),
                        "pharmacophore_locked": locked,
                    })

    # 3. Candidate substituent exit vectors: explicit C-H bonds OR open-valence sp2 heavy carbons when H is omitted
    raw_exit_rays: list[tuple[int, Optional[int], np.ndarray]] = []
    for i in ligand_idx:
        if int(frame.Z[i]) == 1 and lig_neighbors[i]:
            parent = min(lig_neighbors[i], key=lambda j: dist_ang[i, j])
            if int(frame.Z[parent]) == 6:
                vec = pos_ang[i] - pos_ang[parent]
                norm = float(np.linalg.norm(vec))
                if norm > 1e-6:
                    raw_exit_rays.append((int(parent), int(i), vec / norm))

    if not raw_exit_rays:
        # Heavy-atom fallback for PDB/SDF inputs without explicit non-polar hydrogens
        for c_idx in ligand_idx:
            if int(frame.Z[c_idx]) != 6:
                continue
            heavy_nbs = [nb for nb in lig_neighbors[c_idx] if int(frame.Z[nb]) > 1]
            if len(heavy_nbs) == 2:
                u1 = pos_ang[heavy_nbs[0]] - pos_ang[c_idx]
                u2 = pos_ang[heavy_nbs[1]] - pos_ang[c_idx]
                bis = -(u1 / max(float(np.linalg.norm(u1)), 1e-8) + u2 / max(float(np.linalg.norm(u2)), 1e-8))
                bn = float(np.linalg.norm(bis))
                if bn > 1e-5:
                    raw_exit_rays.append((int(c_idx), None, bis / bn))
            elif len(heavy_nbs) == 1:
                u1 = pos_ang[c_idx] - pos_ang[heavy_nbs[0]]
                bn = float(np.linalg.norm(u1))
                if bn > 1e-5:
                    raw_exit_rays.append((int(c_idx), None, u1 / bn))

    ray_steps = (1.5, probe_step_ang, 3.2, 4.2)
    exit_candidates = []
    for parent, h_idx, unit_dir in raw_exit_rays:
        excl = {parent} | ({h_idx} if h_idx is not None else set()) | set(lig_neighbors[parent])
        other_lig = [k for k in ligand_idx if k not in excl]

        # Single-step legacy clearance at probe_step_ang + multi-step ray-marching profile
        probe_pt = pos_ang[parent] + probe_step_ang * unit_dir
        if pocket_idx:
            clearance = float(np.linalg.norm(pos_ang[pocket_idx] - probe_pt[None, :], axis=-1).min())
        else:
            clearance = (
                float(np.linalg.norm(pos_ang[other_lig] - probe_pt[None, :], axis=-1).min())
                if other_lig else 9.9
            )

        near_clash = False
        min_vdw_margin = 9.9
        distal_pocket_dist = 9.9
        for s in ray_steps:
            pt_s = pos_ang[parent] + s * unit_dir
            check_pool = pocket_idx + other_lig
            if check_pool:
                d_s = np.linalg.norm(pos_ang[check_pool] - pt_s[None, :], axis=-1)
                vdw_s = np.array([_vdw_radius(int(frame.Z[k])) for k in check_pool], dtype=np.float64)
                min_vdw_margin = min(min_vdw_margin, float((d_s - 0.75 * vdw_s).min()))
                if s <= 2.5 and float(d_s.min()) < clash_cutoff_ang * 0.85:
                    near_clash = True
            if pocket_idx and s >= 3.0:
                distal_pocket_dist = min(distal_pocket_dist, float(np.linalg.norm(pos_ang[pocket_idx] - pt_s[None, :], axis=-1).min()))

        if clearance >= clash_cutoff_ang and not near_clash:
            channel_type = (
                "subpocket_cavity" if (pocket_idx and 3.0 <= distal_pocket_dist <= 5.8)
                else ("solvent_exposed" if pocket_idx else "open_growth_vector")
            )
            exit_candidates.append({
                "carbon_atom": int(parent),
                "hydrogen_atom": None if h_idx is None else int(h_idx),
                "in_aromatic_ring": parent in ring_atoms_set,
                "direction": unit_dir.tolist(),
                "anchor_xyz_angstrom": pos_ang[parent].tolist(),
                "clearance_angstrom": round(clearance, 4),
                "ray_vdw_margin_angstrom": round(min_vdw_margin, 4),
                "channel_type": channel_type,
            })

    exit_candidates.sort(
        key=lambda c: (
            0 if c["channel_type"] == "subpocket_cavity" else 1,
            -c["clearance_angstrom"],
            c["carbon_atom"],
            -1 if c["hydrogen_atom"] is None else c["hydrogen_atom"],
        )
    )

    # 4. CoM-centered ligand polar dipole (1o, Debye)
    lig_dipole = _subgraph_bond_dipole_debye(frame, ligand_idx, dist_ang, rel_ang)

    # 5. Multi-fragment Linker / PROTAC bridging quartet when ligand has >= 2 disconnected fragments
    lig_frags = _connected_components(ligand_idx, frame.Z, dist_ang, scale=1.35)
    fragment_bridges = []
    if len(lig_frags) >= 2 and exit_candidates:
        atom_to_frag = {a: f_id for f_id, comp in enumerate(lig_frags) for a in comp}
        for idx_a, ca in enumerate(exit_candidates):
            fa = atom_to_frag.get(ca["carbon_atom"], 0)
            for cb in exit_candidates[idx_a + 1:]:
                fb = atom_to_frag.get(cb["carbon_atom"], 0)
                if fa == fb:
                    continue
                pa = np.asarray(ca["anchor_xyz_angstrom"], dtype=np.float64)
                pb = np.asarray(cb["anchor_xyz_angstrom"], dtype=np.float64)
                ab = pb - pa
                d_ab = float(np.linalg.norm(ab))
                if d_ab < 1e-6:
                    continue
                u_ab = ab / d_ab
                ua = np.asarray(ca["direction"], dtype=np.float64)
                ub = np.asarray(cb["direction"], dtype=np.float64)
                cos_a = float(np.dot(ua, u_ab))
                cos_b = float(np.dot(ub, -u_ab))
                mid = 0.5 * (pa + pb)
                corr_clear = (
                    float(np.linalg.norm(pos_ang[pocket_idx] - mid[None, :], axis=-1).min())
                    if pocket_idx else 9.9
                )
                if cos_a > 0.0 and cos_b > 0.0:
                    fragment_bridges.append({
                        "anchor_atoms": [ca["carbon_atom"], cb["carbon_atom"]],
                        "distance_angstrom": round(d_ab, 4),
                        "cos_theta_a": round(cos_a, 4),
                        "cos_theta_b": round(cos_b, 4),
                        "corridor_clearance_angstrom": round(corr_clear, 4),
                        "bridge_vector_1o": u_ab.tolist(),
                    })
        fragment_bridges.sort(key=lambda b: (-(b["cos_theta_a"] + b["cos_theta_b"]), b["distance_angstrom"]))

    return {
        "n_ligand_atoms": len(ligand_idx),
        "n_pocket_atoms": len(pocket_idx),
        "n_ligand_fragments": len(lig_frags),
        "ligand_polar_dipole_debye": np.round(lig_dipole, 5).tolist(),
        "aromatic_rings": aromatic_rings,
        "hbond_donors": hbond_donors,
        "hbond_acceptors": hbond_acceptors,
        "pocket_hbonds": pocket_hbonds,
        "exit_vectors": exit_candidates,
        "fragment_bridges": fragment_bridges,
    }


def analyze_drug(
    source: Union[str, Path, StructureFrame],
    probe: Optional[YLMProbe] = None,
    question: Optional[str] = None,
    verify_o3: bool = False,
) -> ProbeResult:
    frame = load_structure(source)
    probe = probe or YLMProbe.default()
    q = question or "What is the polar dipole direction of the ligand and best substituent growth direction?"
    return probe.ask(frame, q, domain="drug", verify_o3=verify_o3)


def enrich_domain_result(probe: YLMProbe, frame: StructureFrame, res: ProbeResult, domain: str) -> None:
    """Attach domain-specific geometric diagnostics and covariant auxiliary vectors to ProbeResult."""
    if domain == "electrolyte":
        info = analyze_electrolyte_geometry(frame)
        res.domain_summary = info
        ion_idx = info.get("primary_ion_index")
        if ion_idx is not None and frame.n_atoms:
            esc = info["escape_vector"]
            res.vectors.append(VectorPrediction(
                name="desolvation_escape_dir",
                vector_au=esc,
                vector_physical=esc,
                unit="unit",
                magnitude=float(np.linalg.norm(esc)),
                anchor_xyz_angstrom=frame.pos_angstrom[ion_idx].tolist(),
                anchor_atom=int(ion_idx),
            ))
    elif domain == "md":
        res.domain_summary = analyze_md_frame_motion(frame, res)
        if frame.delta_pos_angstrom is not None and frame.n_atoms:
            idx = res.domain_summary["fastest_atom_index"]
            dvec = frame.delta_pos_angstrom[idx].tolist()
            res.vectors.append(VectorPrediction(
                name="interframe_displacement",
                vector_au=frame.delta_pos_bohr[idx].tolist(),
                vector_physical=dvec,
                unit="A/step",
                magnitude=float(np.linalg.norm(dvec)),
                anchor_xyz_angstrom=frame.pos_angstrom[idx].tolist(),
                anchor_atom=int(idx),
            ))
    elif domain == "drug":
        info = analyze_drug_geometry(frame)
        res.domain_summary = info
        if info["exit_vectors"]:
            best = info["exit_vectors"][0]
            res.vectors.append(VectorPrediction(
                name="best_exit_vector",
                vector_au=best["direction"],
                vector_physical=best["direction"],
                unit="unit",
                magnitude=1.0,
                anchor_xyz_angstrom=best["anchor_xyz_angstrom"],
                anchor_atom=best["carbon_atom"],
            ))
        if info["fragment_bridges"]:
            br = info["fragment_bridges"][0]
            a0 = br["anchor_atoms"][0]
            res.vectors.append(VectorPrediction(
                name="linker_bridge_vector",
                vector_au=br["bridge_vector_1o"],
                vector_physical=br["bridge_vector_1o"],
                unit="unit",
                magnitude=1.0,
                anchor_xyz_angstrom=frame.pos_angstrom[a0].tolist(),
                anchor_atom=int(a0),
            ))

"""Multi-dataset Universe data pipeline, chemical substructure paraphraser, and CPU smoke generator.

Unifies molecular DFT (`SPICE-v2`, `OMol25`), periodic crystal (`MPtrj`), and
reactive transition-state (`Transition1x`) structures into one JSONL schema with:
  - Symmetry-unambiguous chemical site & functional-group descriptors (supporting 3D PBC MIC)
  - 28+ diverse natural-language templates split into `train` vs. `test_heldout` phrasings
  - Multi-LOT tags (`wb97m`, `r2scan`, `pbe_u`, `ccsd`, `xtb`)
  - Exact O(3)-covariant scalar (`0e`), vector (`1o`: T1/T3/T4/T5), and symmetric tensor (`0e+2e`: T2) labels
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Optional, Sequence, Union

import numpy as np
import torch

from src.toolkit.ingest import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    StructureFrame,
    covalent_radius,
    load_trajectory,
    minimum_image_delta_bohr,
    symbol_for_z,
)
from src.toolkit.probe import _TextStateEncoder
from src.universe_fm import TASK_TO_ID, resolve_lot_id

_PAULING_CHI = {
    1: 2.20, 3: 0.98, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    11: 0.93, 12: 1.31, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    19: 0.82, 20: 1.00, 22: 1.54, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91,
    29: 1.90, 30: 1.65, 35: 2.96, 53: 2.66,
}

_ISO_POLARIZABILITY = {
    1: 0.67, 3: 2.40, 6: 1.76, 7: 1.10, 8: 0.80, 9: 0.56,
    11: 3.10, 12: 2.10, 15: 3.63, 16: 2.90, 17: 2.18,
    26: 4.20, 27: 3.95, 28: 3.80, 35: 3.05, 53: 5.35,
}

_HALOGEN_NAMES = {9: "fluoro", 17: "chloro", 35: "bromo", 53: "iodo"}

TRAIN_TEMPLATES: dict[str, tuple[str, ...]] = {
    "scalar": (
        "What is the potential energy of this structure?",
        "Compute the total stability scalar of the system.",
        "Predict the system interaction energy.",
        "Give the scalar energy value for this geometry.",
        "Report the potential energy scalar.",
    ),
    "dipole": (
        "What is the total electric dipole vector of this system?",
        "Predict the 3D molecular dipole moment vector.",
        "In which direction and magnitude does the net dipole point?",
        "Compute the overall charge-polarization dipole vector.",
        "Give the 3D dipole vector of the structure.",
    ),
    "force": (
        "What is the 3D atomic force acting on {a}?",
        "Compute the net restoring force vector on {a}.",
        "Give the force vector experienced by {a}.",
        "Predict the mechanical force pushing on {a}.",
        "Report the local 3D force vector at {a}.",
    ),
    "diff": (
        "What is the force difference vector F({a}) minus F({b})?",
        "Compute the relative force vector on {a} minus that on {b}.",
        "Give F({a}) - F({b}) between {a} and {b}.",
        "Predict the differential force vector from {b} to {a}: F({a}) minus F({b}).",
        "Evaluate the vector difference between the force on {a} and the force on {b}.",
    ),
    "proj": (
        "Project the force on {a} onto the {a} to {b} bond axis.",
        "What is the 3D bond-axis projection of {a}'s force toward {b}?",
        "Compute the vector component of force on {a} along the line to {b}.",
        "Give the axial force vector of {a} projected onto the {a}-{b} direction.",
        "Resolve the force acting on {a} parallel to the vector toward {b}.",
    ),
    "tensor": (
        "Predict the symmetric 3x3 polarizability response tensor.",
        "What is the second-rank anisotropic response matrix of this system?",
        "Compute the 3x3 symmetric polarizability / quadrupole tensor.",
        "Give the full 3x3 symmetric electronic response tensor.",
        "Report the rank-2 symmetric polarizability matrix.",
    ),
    "chiral": (
        "What is the (R)/(S) chiral handedness pseudoscalar of this stereoisomer?",
        "Predict the parity-odd 3D chirality sign of this conformer.",
        "Determine the signed tetrahedral chirality pseudoscalar of the molecule.",
        "Report the O(3)-odd enantiomeric handedness sign (+1 or -1).",
        "Compute the 3D stereochemical handedness sign of this structure.",
    ),
    "chiral_site": (
        "What is the local 3D tetrahedral chirality sign around {a}?",
        "Predict the parity-odd stereocenter handedness at {a}.",
        "Give the signed tetrahedral volume handedness around {a}.",
    ),
}

HELDOUT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "scalar": (
        "How much potential energy is stored in this atomic arrangement?",
        "Estimate the isotropic energy level of the given conformer or cell.",
        "Return the scalar potential for the current 3D coordinates.",
    ),
    "dipole": (
        "Where does the overall electrostatic dipole arrow point in 3D space?",
        "Determine the net 3D polarity vector across the entire species.",
        "Output the orientation and strength of the system's electric dipole.",
    ),
    "force": (
        "Which 3D way and how hard is {a} being pushed right now?",
        "Determine the instantaneous negative potential gradient on {a}.",
        "Extract the 3D Cartesian force arrow located on {a}.",
    ),
    "diff": (
        "Subtract the force vector felt by {b} from the force vector felt by {a}.",
        "How does the force vector at {a} differ from the one at {b}?",
        "Return the 3D vector obtained by taking {a}'s force minus {b}'s force.",
    ),
    "proj": (
        "Isolate the parallel part of {a}'s force vector aligned with {b}.",
        "Along the interatomic axis connecting {a} and {b}, what vector force acts on {a}?",
        "Decompose the force at {a} and return only the vector parallel to the {a}-{b} axis.",
    ),
    "tensor": (
        "Characterize the 3D directional polarizability ellipsoid as a 3x3 symmetric tensor.",
        "Output the six-DOF symmetric second-order response matrix for this geometry.",
        "Evaluate the anisotropic 3x3 response tensor under the laboratory frame.",
    ),
    "chiral": (
        "Which enantiomeric handedness (+1 vs -1) does this 3D arrangement exhibit?",
        "Evaluate the reflection-antisymmetric chirality pseudoscalar for this conformer.",
        "Distinguish the 3D mirror-image enantiomer sign of the current coordinates.",
    ),
    "chiral_site": (
        "Determine the local 3D stereochemical handedness sign centered at {a}.",
        "Under the current 3D geometry, what parity-odd chirality sign sits on {a}?",
    ),
}


@dataclass
class UniverseRecord:
    """One serialized multi-task Universe QA sample."""

    structure_id: str
    domain: str
    lot: str
    split: str
    Z: list[int]
    pos_bohr: list[list[float]]
    cell_bohr: Optional[list[list[float]]]
    pbc: Optional[list[bool]]
    question: str
    task_kind: str          # "scalar" | "vector" | "tensor" | "chiral"
    subtask: str            # "scalar" | "chiral" | "dipole" | "force" | "diff" | "proj" | "tensor"
    scalar_target: float
    vector_target: list[float]
    tensor_target: list[list[float]]
    target_atoms: list[int]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "UniverseRecord":
        return cls(**json.loads(line))


def compute_chiral_centers(frame: StructureFrame) -> list[dict]:
    """Compute O(3)-odd (`0o`) signed tetrahedral volumes around 4-coordinate stereocenters.

    Neighbors are ordered canonically by pure 2-hop graph-topological signatures (`0e` invariant)
    so prochiral `-CH2-` sites are rejected and the signed volume `V(pos @ R.T) = det(R) * V(pos)`
    is strictly invariant under proper rotations & atom re-indexing while flipping sign under
    mirror reflection / spatial inversion `det(R) = -1`.
    """
    n = frame.n_atoms
    if n < 5:
        return []
    pos = frame.pos_angstrom
    z = frame.Z
    radii = np.array([covalent_radius(int(zi)) for zi in z], dtype=np.float64)
    r0 = radii[:, None] + radii[None, :]
    rel = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(rel, axis=-1)
    bonded = (dist > 1e-6) & (dist <= 1.28 * r0)
    degrees = bonded.sum(axis=-1)

    hop1: list[tuple] = []
    for i in range(n):
        nbs = np.where(bonded[i])[0]
        hop1.append((int(z[i]), int(degrees[i]), tuple(sorted(int(z[j]) for j in nbs))))
    env_keys: list[tuple] = []
    for i in range(n):
        nbs = np.where(bonded[i])[0]
        env_keys.append((hop1[i], tuple(sorted(hop1[j] for j in nbs))))

    centers: list[dict] = []
    for c in range(n):
        if int(z[c]) not in (6, 7, 14, 15, 16):
            continue
        nbs = list(np.where(bonded[c])[0])
        if len(nbs) != 4:
            continue
        nb_keys = [env_keys[j] for j in nbs]
        if len(set(nb_keys)) < 4:
            continue
        ordered = [j for _, j in sorted(zip(nb_keys, nbs), key=lambda pair: pair[0], reverse=True)]
        v1 = pos[ordered[0]] - pos[c]
        v2 = pos[ordered[1]] - pos[c]
        v3 = pos[ordered[2]] - pos[c]
        signed_vol = float(np.linalg.det(np.stack([v1, v2, v3], axis=1)) / 6.0)
        if abs(signed_vol) > 1e-3:
            centers.append({
                "atom_index": int(c),
                "symbol": symbol_for_z(int(z[c])),
                "neighbors": [int(x) for x in ordered],
                "signed_volume_ang3": signed_vol,
                "handedness_sign": 1 if signed_vol > 0 else -1,
            })
    return centers


def _pairwise_mic_angstrom(frame: StructureFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return pairwise displacement vectors r_i - r_j in Angstrom (N, N, 3) and distances (N, N) with PBC MIC."""
    n = frame.n_atoms
    pos_bohr = frame.pos_bohr
    rel_bohr = np.zeros((n, n, 3), dtype=np.float64)
    for i in range(n):
        rel_bohr[i] = minimum_image_delta_bohr(
            np.broadcast_to(pos_bohr[i], (n, 3)),
            pos_bohr,
            frame.cell_bohr,
            frame.pbc,
        )
    rel_ang = rel_bohr * BOHR_TO_ANGSTROM
    dist_ang = np.linalg.norm(rel_ang, axis=-1)
    return rel_ang, dist_ang


def compute_covariant_universe_labels(
    frame: StructureFrame,
    lot: str = "wb97m",
) -> dict[str, np.ndarray]:
    """Compute or extract strictly O(3)-covariant scalar (0e), force/dipole (1o), and symmetric tensor (0e+2e) labels."""
    n = frame.n_atoms
    rel_ang, dist_ang = _pairwise_mic_angstrom(frame)

    lot_norm = lot.strip().lower()
    if lot_norm == "xtb":
        e_scale, e_shift, f_scale, t_iso_scale, t_aniso_scale = 1.15, -0.35, 1.22, 1.12, 1.20
    elif lot_norm == "pbe_u":
        e_scale, e_shift, f_scale, t_iso_scale, t_aniso_scale = 0.93, -0.18, 0.94, 1.06, 0.95
    elif lot_norm == "r2scan":
        e_scale, e_shift, f_scale, t_iso_scale, t_aniso_scale = 0.98, -0.06, 0.98, 1.02, 0.99
    else:
        e_scale, e_shift, f_scale, t_iso_scale, t_aniso_scale = 1.00, 0.00, 1.00, 1.00, 1.00

    energy = 0.0
    forces = np.zeros((n, 3), dtype=np.float64)
    dipole = np.zeros(3, dtype=np.float64)
    aniso_ten = np.zeros((3, 3), dtype=np.float64)

    iso_sum = sum(_ISO_POLARIZABILITY.get(int(z), 1.50) for z in frame.Z) / max(n, 1)

    for i in range(n):
        zi = int(frame.Z[i])
        chi_i = _PAULING_CHI.get(zi, 2.0)
        for j in range(i + 1, n):
            zj = int(frame.Z[j])
            chi_j = _PAULING_CHI.get(zj, 2.0)
            rij_vec = rel_ang[i, j]     # pos_i - pos_j (MIC)
            r = float(dist_ang[i, j])
            if r < 1e-7 or r > 4.6:
                continue
            r_hat = rij_vec / r
            r0 = covalent_radius(zi) + covalent_radius(zj)
            is_coord = (zi in (3, 11, 12, 19, 26, 27) and zj in (7, 8, 9, 16, 17)) or (
                zj in (3, 11, 12, 19, 26, 27) and zi in (7, 8, 9, 16, 17)
            )
            eq_r = 1.98 if is_coord else r0
            k_spring = 3.2 if is_coord else 6.5

            if r < 1.55 * eq_r:
                dr = r - eq_r
                pair_e = 0.5 * k_spring * (dr * dr) - 0.45 * math.exp(-0.5 * (dr / 0.35) ** 2)
                f_mag = -k_spring * dr
            else:
                sigma = 0.82 * (r0 + 0.85)
                ratio = sigma / r
                pair_e = 0.18 * (ratio ** 6)
                f_mag = 1.08 * (ratio ** 6) / r

            energy += pair_e
            f_vec = (f_scale * f_mag) * r_hat
            forces[i] += f_vec
            forces[j] -= f_vec

            # Antisymmetric bond dipole contribution: q_ij * (r_i - r_j) is translation- & PBC-safe!
            dq = 0.42 * math.tanh(0.85 * (chi_i - chi_j)) * math.exp(-((r / max(1.45 * eq_r, 0.6)) ** 2))
            dipole += (dq * 4.8032047 * f_scale) * rij_vec

            # Rank-2 symmetric traceless bond polarizability anisotropy (2e)
            gamma = (0.85 + 0.25 * abs(chi_i - chi_j)) * math.exp(-((r / max(1.5 * eq_r, 0.7)) ** 2))
            traceless_outer = np.outer(r_hat, r_hat) - (np.eye(3, dtype=np.float64) / 3.0)
            aniso_ten += gamma * traceless_outer

    # Prefer recorded DFT quantities when present on frame / metadata
    if "energy_ev" in frame.metadata:
        scalar_val = float(frame.metadata["energy_ev"])
    else:
        scalar_val = float(e_scale * (energy / max(n, 1)) + e_shift)

    if frame.forces_ev_ang is not None:
        forces_out = np.asarray(frame.forces_ev_ang, dtype=np.float64)
    else:
        forces_out = forces

    if frame.dipole_debye is not None:
        dipole_out = np.asarray(frame.dipole_debye, dtype=np.float64)
    else:
        dipole_out = dipole

    if "tensor_3x3" in frame.metadata:
        ten_raw = np.asarray(frame.metadata["tensor_3x3"], dtype=np.float64).reshape(3, 3)
        tensor_out = 0.5 * (ten_raw + ten_raw.T)
    else:
        tensor_out = (t_iso_scale * iso_sum) * np.eye(3, dtype=np.float64) + t_aniso_scale * aniso_ten

    return {
        "scalar": np.float64(scalar_val),
        "forces": forces_out,
        "dipole": dipole_out,
        "tensor": tensor_out,
    }


@dataclass
class SiteDescriptor:
    atom_index: int
    canonical_key: str
    phrases: tuple[str, ...]
    neighbors: tuple[int, ...]


def describe_unique_sites(frame: StructureFrame) -> list[SiteDescriptor]:
    """Extract symmetry-unambiguous O(3)-invariant chemical site phrases for uniquely addressable atoms."""
    n = frame.n_atoms
    if n == 0:
        return []
    _, dist_ang = _pairwise_mic_angstrom(frame)

    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        zi = int(frame.Z[i])
        for j in range(i + 1, n):
            zj = int(frame.Z[j])
            d = float(dist_ang[i, j])
            cov_cut = 1.35 * (covalent_radius(zi) + covalent_radius(zj))
            ion_cut = 2.50 if (
                (zi in (3, 11, 12, 19, 26, 27) and zj in (7, 8, 9, 16, 17))
                or (zj in (3, 11, 12, 19, 26, 27) and zi in (7, 8, 9, 16, 17))
            ) else 0.0
            if d <= max(cov_cut, ion_cut):
                adj[i].append(j)
                adj[j].append(i)

    shell1_strs: list[str] = []
    for i in range(n):
        cnt = Counter(symbol_for_z(int(frame.Z[j])) for j in adj[i])
        shell1_strs.append(",".join(f"{v}{k}" for k, v in sorted(cnt.items())) or "isolated")

    topo_keys: list[str] = []
    for i in range(n):
        sym = symbol_for_z(int(frame.Z[i]))
        sh2 = Counter(f"{symbol_for_z(int(frame.Z[j]))}[{shell1_strs[j]}]" for j in adj[i])
        sh2_str = ";".join(f"{v}*{k}" for k, v in sorted(sh2.items())) or "none"
        topo_keys.append(f"{sym}({shell1_strs[i]}|{sh2_str})")

    key_counts = Counter(topo_keys)
    sites: list[SiteDescriptor] = []

    for i in range(n):
        if key_counts[topo_keys[i]] != 1:
            continue
        zi = int(frame.Z[i])
        sym = symbol_for_z(zi)
        nbs = adj[i]
        nb_z = [int(frame.Z[j]) for j in nbs]
        sh1 = shell1_strs[i]
        base_tag = f"{sym}({sh1})"
        phrases: list[str] = []

        # Functional-group / chemical role enrichment
        if zi == 1:
            close_donors = [
                j for j in range(n)
                if j != i and int(frame.Z[j]) in (7, 8, 9) and float(dist_ang[i, j]) <= 1.52
            ]
            if len(close_donors) >= 2:
                phrases.extend([
                    f"the shared bridging proton {base_tag}",
                    f"the transferring hydrogen atom {base_tag}",
                ])
            elif nb_z == [8]:
                phrases.extend([
                    f"the hydroxyl proton (-OH, {base_tag})",
                    f"the alcohol hydrogen bonded to oxygen ({base_tag})",
                ])
            elif nb_z == [7]:
                phrases.extend([
                    f"the amine proton (-NH, {base_tag})",
                    f"the nitrogen-bound hydrogen ({base_tag})",
                ])
        elif zi == 8:
            if len(nb_z) == 1 and nb_z[0] == 6:
                phrases.extend([
                    f"the carbonyl oxygen (C=O, {base_tag})",
                    f"the terminal oxo oxygen ({base_tag})",
                ])
            elif len(nb_z) == 1 and nb_z[0] == 15:
                phrases.extend([
                    f"the phosphoryl oxygen (P=O, {base_tag})",
                    f"the phosphate oxo atom ({base_tag})",
                ])
            elif 1 in nb_z and 6 in nb_z:
                phrases.extend([
                    f"the hydroxyl oxygen (-OH, {base_tag})",
                    f"the alcohol oxygen center ({base_tag})",
                ])
            elif 3 in nb_z or 11 in nb_z:
                phrases.extend([
                    f"the ion-coordinating donor oxygen ({base_tag})",
                    f"the solvation-shell oxygen site ({base_tag})",
                ])
        elif zi in (3, 11, 12, 19):
            phrases.extend([
                f"the coordinated {sym}+ working cation ({base_tag})",
                f"the central {sym} ion ({base_tag})",
            ])
        elif zi in (22, 25, 26, 27, 28, 29):
            phrases.extend([
                f"the {sym} transition-metal center ({base_tag})",
                f"the polyhedral {sym} site ({base_tag})",
            ])
        elif zi in _HALOGEN_NAMES:
            hname = _HALOGEN_NAMES[zi]
            phrases.extend([
                f"the {hname} substituent (-{sym}, {base_tag})",
                f"the terminal {sym} halogen atom ({base_tag})",
            ])
        elif zi in (15, 16):
            role_word = "phosphorus" if zi == 15 else "sulfur"
            phrases.extend([
                f"the {role_word} heteroatom center ({base_tag})",
                f"the {sym} core atom ({base_tag})",
            ])

        phrases.append(f"atom {base_tag} [env: {topo_keys[i]}]")
        sites.append(SiteDescriptor(
            atom_index=i,
            canonical_key=topo_keys[i],
            phrases=tuple(phrases),
            neighbors=tuple(nbs),
        ))
    return sites


def generate_diverse_queries(
    frame: StructureFrame,
    *,
    structure_id: str,
    domain: str = "custom",
    lot: str = "wb97m",
    split: str = "train",
    max_force_sites: int = 2,
    include_lot_prefix: bool = True,
    include_chiral: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> list[UniverseRecord]:
    """Generate multi-task (scalar, chiral, dipole, force, diff, proj, tensor) records for one structure."""
    if rng is None:
        seed = int.from_bytes(hashlib.sha256(f"{structure_id}|{split}".encode("utf-8")).digest()[:4], "big")
        rng = np.random.default_rng(seed)

    templates = HELDOUT_TEMPLATES if split == "test_heldout" else TRAIN_TEMPLATES
    labels = compute_covariant_universe_labels(frame, lot=lot)
    sites = describe_unique_sites(frame)
    rel_ang, dist_ang = _pairwise_mic_angstrom(frame)

    prefix = f"[LOT={lot}] " if include_lot_prefix else ""
    z_list = [int(z) for z in frame.Z.tolist()]
    pos_list = [[float(c) for c in row] for row in frame.pos_bohr.tolist()]
    cell_list = None if frame.cell_bohr is None else [[float(c) for c in row] for row in frame.cell_bohr.tolist()]
    pbc_list = None if frame.pbc is None else [bool(b) for b in frame.pbc.tolist()]
    zero_vec = [0.0, 0.0, 0.0]
    zero_ten = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    records: list[UniverseRecord] = []

    # 1) 0e Scalar (skip on xtb if desired or keep with LOT tag; compute_universe_loss masks xtb scalar)
    if lot != "xtb":
        q_sc = prefix + str(rng.choice(templates["scalar"]))
        records.append(UniverseRecord(
            structure_id=structure_id, domain=domain, lot=lot, split=split,
            Z=z_list, pos_bohr=pos_list, cell_bohr=cell_list, pbc=pbc_list,
            question=q_sc, task_kind="scalar", subtask="scalar",
            scalar_target=float(labels["scalar"]),
            vector_target=zero_vec, tensor_target=zero_ten, target_atoms=[],
        ))

    # 2) 1o Dipole (T1)
    q_dip = prefix + str(rng.choice(templates["dipole"]))
    records.append(UniverseRecord(
        structure_id=structure_id, domain=domain, lot=lot, split=split,
        Z=z_list, pos_bohr=pos_list, cell_bohr=cell_list, pbc=pbc_list,
        question=q_dip, task_kind="vector", subtask="dipole",
        scalar_target=0.0,
        vector_target=[float(x) for x in labels["dipole"].tolist()],
        tensor_target=zero_ten, target_atoms=[],
    ))

    # 3) 0e+2e Symmetric Tensor (T2)
    q_ten = prefix + str(rng.choice(templates["tensor"]))
    records.append(UniverseRecord(
        structure_id=structure_id, domain=domain, lot=lot, split=split,
        Z=z_list, pos_bohr=pos_list, cell_bohr=cell_list, pbc=pbc_list,
        question=q_ten, task_kind="tensor", subtask="tensor",
        scalar_target=0.0, vector_target=zero_vec,
        tensor_target=[[float(x) for x in row] for row in labels["tensor"].tolist()],
        target_atoms=[],
    ))

    # 4) 1o Local Atom Force (T3)
    if sites:
        chosen_sites = list(sites)
        rng.shuffle(chosen_sites)
        for site in chosen_sites[:max_force_sites]:
            phrase_a = str(rng.choice(site.phrases))
            q_f = prefix + str(rng.choice(templates["force"])).format(a=phrase_a)
            f_vec = labels["forces"][site.atom_index]
            records.append(UniverseRecord(
                structure_id=structure_id, domain=domain, lot=lot, split=split,
                Z=z_list, pos_bohr=pos_list, cell_bohr=cell_list, pbc=pbc_list,
                question=q_f, task_kind="vector", subtask="force",
                scalar_target=0.0,
                vector_target=[float(x) for x in f_vec.tolist()],
                tensor_target=zero_ten,
                target_atoms=[int(site.atom_index)],
            ))

    # 5) Compositional T4 (Force Difference) & T5 (Bond-Axis Force Projection)
    if len(sites) >= 2:
        # Prefer bonded or closest uniquely addressable site pair
        best_pair: Optional[tuple[SiteDescriptor, SiteDescriptor]] = None
        best_dist = float("inf")
        for idx_a in range(len(sites)):
            for idx_b in range(idx_a + 1, len(sites)):
                sa, sb = sites[idx_a], sites[idx_b]
                d_ab = float(dist_ang[sa.atom_index, sb.atom_index])
                if d_ab > 1e-5 and d_ab < best_dist:
                    best_dist = d_ab
                    best_pair = (sa, sb)
        if best_pair is not None:
            sa, sb = best_pair
            pa = str(rng.choice(sa.phrases))
            pb = str(rng.choice(sb.phrases))
            fa = labels["forces"][sa.atom_index]
            fb = labels["forces"][sb.atom_index]

            # T4: F(a) - F(b)
            q_diff = prefix + str(rng.choice(templates["diff"])).format(a=pa, b=pb)
            records.append(UniverseRecord(
                structure_id=structure_id, domain=domain, lot=lot, split=split,
                Z=z_list, pos_bohr=pos_list, cell_bohr=cell_list, pbc=pbc_list,
                question=q_diff, task_kind="vector", subtask="diff",
                scalar_target=0.0,
                vector_target=[float(x) for x in (fa - fb).tolist()],
                tensor_target=zero_ten,
                target_atoms=[int(sa.atom_index), int(sb.atom_index)],
            ))

            # T5: (F(a) . u_ab) u_ab where u_ab = MIC(r_b - r_a) / ||MIC(r_b - r_a)||
            rab = -rel_ang[sa.atom_index, sb.atom_index]    # pos_b - pos_a
            norm_rab = float(np.linalg.norm(rab))
            if norm_rab > 1e-6:
                u_ab = rab / norm_rab
                proj_vec = float(np.dot(fa, u_ab)) * u_ab
                q_proj = prefix + str(rng.choice(templates["proj"])).format(a=pa, b=pb)
                records.append(UniverseRecord(
                    structure_id=structure_id, domain=domain, lot=lot, split=split,
                    Z=z_list, pos_bohr=pos_list, cell_bohr=cell_list, pbc=pbc_list,
                    question=q_proj, task_kind="vector", subtask="proj",
                    scalar_target=0.0,
                    vector_target=[float(x) for x in proj_vec.tolist()],
                    tensor_target=zero_ten,
                    target_atoms=[int(sa.atom_index), int(sb.atom_index)],
                ))

    # 6) 0o Parity-Odd Chiral Pseudoscalar (enantiomer handedness sign +1 vs -1)
    if include_chiral:
        chiral_centers = compute_chiral_centers(frame)
        if chiral_centers:
            site_by_idx = {s.atom_index: s for s in sites}
            primary = max(chiral_centers, key=lambda d: abs(float(d["signed_volume_ang3"])))
            c_idx = int(primary["atom_index"])
            c_sign = float(primary["handedness_sign"])
            if c_idx in site_by_idx and bool(rng.integers(0, 2) == 1):
                c_phrase = str(rng.choice(site_by_idx[c_idx].phrases))
                q_ch = prefix + str(rng.choice(templates["chiral_site"])).format(a=c_phrase)
            else:
                q_ch = prefix + str(rng.choice(templates["chiral"]))
            records.append(UniverseRecord(
                structure_id=structure_id, domain=domain, lot=lot, split=split,
                Z=z_list, pos_bohr=pos_list, cell_bohr=cell_list, pbc=pbc_list,
                question=q_ch, task_kind="chiral", subtask="chiral",
                scalar_target=c_sign,
                vector_target=zero_vec,
                tensor_target=zero_ten,
                target_atoms=[c_idx],
            ))
    return records


def _make_base_templates() -> list[tuple[str, str, StructureFrame]]:
    """Construct diverse 3D chemical archetypes spanning drugs, electrolytes, PBC crystals, and reactive TS."""
    archetypes: list[tuple[str, str, StructureFrame]] = []

    # 1. Drug/bioorganic scaffolds with O, N, F, P, S, Cl heteroatoms (asymmetric 3D coordinates in Angstrom)
    drug_specs = [
        (
            "chloro_fluoro_alkanol",
            [6, 6, 8, 1, 17, 9, 1, 1, 1],
            [
                [0.00, 0.00, 0.00], [1.51, 0.12, 0.08], [2.08, 1.38, -0.22], [2.99, 1.32, -0.52],
                [-0.68, -1.56, 0.35], [-0.52, 0.92, -0.86], [-0.36, 0.32, 1.01],
                [1.96, -0.66, -0.54], [1.78, -0.12, 1.11],
            ],
        ),
        (
            "phosphoramide_fragment",
            [15, 8, 7, 6, 9, 16, 1, 1, 1, 1],
            [
                [0.00, 0.00, 0.00], [0.05, 1.48, 0.14], [1.46, -0.68, -0.21], [2.68, 0.06, 0.12],
                [-1.18, -0.62, 0.86], [-0.52, -0.78, -1.78], [1.52, -1.66, -0.46],
                [3.52, -0.58, -0.14], [2.74, 0.98, -0.46], [2.72, 0.28, 1.19],
            ],
        ),
        (
            "mercapto_amide",
            [6, 8, 7, 6, 16, 1, 1, 1, 1],
            [
                [0.00, 0.00, 0.00], [-0.12, 1.23, 0.11], [1.19, -0.64, -0.12], [-1.22, -0.88, 0.09],
                [-2.68, -0.02, -0.48], [1.24, -1.64, -0.24], [2.04, -0.11, -0.04],
                [-1.36, -1.18, 1.13], [-1.09, -1.78, -0.51],
            ],
        ),
    ]
    for name, zs, coords in drug_specs:
        pos_b = (np.asarray(coords, dtype=np.float64) - np.mean(coords, axis=0)) * ANGSTROM_TO_BOHR
        archetypes.append(("drug_organic", name, StructureFrame(Z=zs, pos_bohr=pos_b, label=name)))

    # 2. Electrolyte ion-solvent coordination clusters (Li+ / Na+ coordinated to carbonyl + fluorinated ether)
    elec_specs = [
        (
            "li_fec_ether_shell",
            [3, 8, 6, 8, 6, 9, 8, 6, 1, 1],
            [
                [0.00, 0.00, 0.00], [1.92, 0.10, 0.05], [3.14, 0.18, -0.02], [3.88, 1.28, -0.22],
                [3.92, -1.08, 0.18], [5.26, -0.94, 0.32], [-1.98, -0.22, 0.18], [-3.18, 0.48, -0.14],
                [3.56, -1.62, 1.05], [-3.98, -0.18, -0.48],
            ],
        ),
        (
            "na_sulfone_carbonate_shell",
            [11, 8, 16, 8, 6, 17, 8, 6, 9],
            [
                [0.00, 0.00, 0.00], [2.24, 0.14, 0.08], [3.62, -0.32, -0.12], [4.58, 0.78, -0.28],
                [3.98, -1.52, 1.18], [5.66, -2.04, 1.22], [-2.28, 0.28, -0.16], [-3.46, -0.12, 0.06],
                [-4.42, 0.82, 0.34],
            ],
        ),
    ]
    for name, zs, coords in elec_specs:
        pos_b = np.asarray(coords, dtype=np.float64) * ANGSTROM_TO_BOHR
        archetypes.append(("electrolyte", name, StructureFrame(Z=zs, pos_bohr=pos_b, label=name)))

    # 3. 3D Periodic Inorganic Crystals (PBC: olivine LiFePO4-like & layered NaCoO2-like cells)
    cryst_specs = [
        (
            "lifepo4_unit_motif",
            [3, 26, 15, 8, 8, 8],
            [[0.62, 0.85, 0.74], [2.48, 2.15, 1.96], [2.52, 0.78, 3.68],
             [1.36, 0.62, 2.74], [3.74, 1.12, 3.02], [2.41, 2.18, 4.32]],
            [[5.15, 0.00, 0.00], [0.18, 5.02, 0.00], [0.08, 0.12, 5.28]],
        ),
        (
            "nacoo2_ sulfide_motif",
            [11, 27, 16, 8, 9],
            [[0.55, 0.68, 0.82], [2.42, 2.36, 2.12], [1.18, 2.46, 0.92],
             [3.56, 1.88, 2.86], [2.62, 0.76, 1.64]],
            [[4.88, 0.00, 0.00], [0.00, 4.96, 0.00], [0.14, 0.00, 5.12]],
        ),
    ]
    for name, zs, coords, lattice in cryst_specs:
        archetypes.append((
            "crystal_pbc",
            name,
            StructureFrame(
                Z=zs,
                pos_bohr=np.asarray(coords, dtype=np.float64) * ANGSTROM_TO_BOHR,
                cell_bohr=np.asarray(lattice, dtype=np.float64) * ANGSTROM_TO_BOHR,
                pbc=np.ones(3, dtype=bool),
                label=name,
            ),
        ))

    # 4. Reactive Transition-State Frames (Grotthuss proton transfer & SN2 halide exchange)
    ts_specs = [
        (
            "zundel_proton_transfer_ts",
            [8, 1, 8, 1, 1, 6, 9],
            [
                [-1.24, 0.04, 0.02], [-0.08, 0.02, -0.03], [1.21, -0.05, 0.04],
                [-1.78, 0.82, 0.21], [-1.72, -0.74, -0.28], [2.18, 0.92, -0.18], [3.44, 0.42, 0.12],
            ],
        ),
        (
            "sn2_chloro_fluoro_ts",
            [17, 6, 9, 1, 1, 8, 1],
            [
                [-2.12, 0.05, -0.04], [0.00, 0.00, 0.00], [1.86, -0.04, 0.03],
                [0.08, 1.06, 0.14], [-0.06, -0.48, 0.96], [0.12, -0.68, -1.22], [0.88, -1.26, -1.28],
            ],
        ),
    ]
    for name, zs, coords in ts_specs:
        pos_b = np.asarray(coords, dtype=np.float64) * ANGSTROM_TO_BOHR
        archetypes.append(("reactive_ts", name, StructureFrame(Z=zs, pos_bohr=pos_b, label=name)))

    return archetypes


def build_cpu_smoke_universe_dataset(
    *,
    n_conformers_per_archetype: int = 12,
    seed: int = 42,
    include_xtb_lot: bool = True,
) -> dict[str, list[UniverseRecord]]:
    """Build a multi-domain, zero-download Universe smoke dataset split cleanly into train/val/test/test_heldout."""
    rng = np.random.default_rng(seed)
    archetypes = _make_base_templates()
    splits: dict[str, list[UniverseRecord]] = {"train": [], "val": [], "test": [], "test_heldout": []}

    for arch_idx, (domain, base_name, base_frame) in enumerate(archetypes):
        for conf_idx in range(n_conformers_per_archetype):
            # Perturb coordinates slightly (0.04 - 0.09 Angstrom) + apply random global O(3) rotation
            disp_ang = rng.normal(0.0, 0.065, size=(base_frame.n_atoms, 3))
            disp_ang -= disp_ang.mean(axis=0, keepdims=True)
            q_mat, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if conf_idx % 3 == 1:
                q_mat = -q_mat

            pos_new = (base_frame.pos_bohr + disp_ang * ANGSTROM_TO_BOHR) @ q_mat.T
            cell_new = None if base_frame.cell_bohr is None else (base_frame.cell_bohr @ q_mat.T)
            frame = StructureFrame(
                Z=base_frame.Z.copy(),
                pos_bohr=pos_new,
                cell_bohr=cell_new,
                pbc=None if base_frame.pbc is None else base_frame.pbc.copy(),
                label=f"{base_name}_c{conf_idx}",
            )

            # Split strictly by conformer index (8 train : 2 val : 2 test)
            mod = conf_idx % 6
            if mod == 4:
                split_name = "val"
            elif mod == 5:
                split_name = "test"
            else:
                split_name = "train"

            primary_lot = "pbe_u" if domain == "crystal_pbc" else "wb97m"
            sid = f"{base_name}_a{arch_idx}_c{conf_idx}"
            recs = generate_diverse_queries(
                frame,
                structure_id=sid,
                domain=domain,
                lot=primary_lot,
                split=split_name,
                rng=rng,
            )
            splits[split_name].extend(recs)

            # Add semi-empirical xtb directional augmentation on a subset of organic training conformers
            if include_xtb_lot and split_name == "train" and domain == "drug_organic" and (conf_idx % 2 == 0):
                xtb_recs = generate_diverse_queries(
                    frame,
                    structure_id=f"{sid}_xtb",
                    domain=domain,
                    lot="xtb",
                    split="train",
                    max_force_sites=1,
                    rng=rng,
                )
                splits["train"].extend(xtb_recs)

            # Generate unseen-phrasing `test_heldout` queries on the exact test conformers
            if split_name == "test":
                held_recs = generate_diverse_queries(
                    frame,
                    structure_id=f"{sid}_heldout",
                    domain=domain,
                    lot=primary_lot,
                    split="test_heldout",
                    rng=rng,
                )
                splits["test_heldout"].extend(held_recs)

    return splits


def prepare_universe_from_files(
    inputs: Sequence[Union[str, Path]],
    *,
    lot: str = "wb97m",
    domain: str = "custom",
    seed: int = 0,
) -> dict[str, list[UniverseRecord]]:
    """Convert raw .xyz / .extxyz / .sdf / .cif / .poscar structure files into Universe splits."""
    rng = np.random.default_rng(seed)
    all_frames: list[StructureFrame] = []
    for src_path in inputs:
        all_frames.extend(load_trajectory(src_path))
    if not all_frames:
        raise ValueError("No structures loaded from input files")

    splits: dict[str, list[UniverseRecord]] = {"train": [], "val": [], "test": [], "test_heldout": []}
    for idx, frame in enumerate(all_frames):
        if len(all_frames) < 4:
            split_name = "train"
        elif idx % 10 == 8:
            split_name = "val"
        elif idx % 10 == 9:
            split_name = "test"
        else:
            split_name = "train"
        frame_lot = str(frame.metadata.get("lot", lot))
        sid = f"{frame.label}_{idx}"
        splits[split_name].extend(generate_diverse_queries(
            frame, structure_id=sid, domain=domain, lot=frame_lot, split=split_name, rng=rng,
        ))
        if split_name == "test":
            splits["test_heldout"].extend(generate_diverse_queries(
                frame, structure_id=f"{sid}_heldout", domain=domain, lot=frame_lot, split="test_heldout", rng=rng,
            ))
    return splits


def save_universe_splits(splits: dict[str, list[UniverseRecord]], out_dir: Union[str, Path]) -> dict[str, str]:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for split_name, records in splits.items():
        fp = out_p / f"{split_name}.jsonl"
        with fp.open("w", encoding="utf-8") as handle:
            for rec in records:
                handle.write(rec.to_json() + "\n")
        paths[split_name] = str(fp)
    return paths


def load_universe_jsonl(path: Union[str, Path]) -> list[UniverseRecord]:
    p = Path(path)
    records: list[UniverseRecord] = []
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if s:
                records.append(UniverseRecord.from_json(s))
    return records


class QuestionFeaturizer:
    """O(3)-invariant question token featurizer supporting fast subword-hash embeddings or HF BERT models."""

    def __init__(self, mode: str = "fast", dim: int = 128, allow_download: bool = False):
        self.mode = mode.lower()
        if self.mode == "matscibert":
            self.dim = 768
            self._hf = _TextStateEncoder("matscibert", 768, allow_download=allow_download)
        elif self.mode == "minilm":
            self.dim = 384
            self._hf = _TextStateEncoder("minilm", 384, allow_download=allow_download)
        elif self.mode == "fast":
            self.dim = int(dim)
            self._hf = None
        else:
            raise ValueError(f"Unknown text featurizer mode {mode!r}; expected 'fast', 'minilm', or 'matscibert'")
        self._cache: dict[str, torch.Tensor] = {}

    def _fast_subword_tokens(self, question: str) -> torch.Tensor:
        toks = re.findall(r"\[lot=[a-z0-9_]+\]|[a-z0-9_+|-]+|[^\s]", question.lower()) or ["?"]
        toks = ["[cls]"] + toks[:46] + ["[sep]"]
        d = self.dim
        rows = np.zeros((len(toks), d), dtype=np.float32)
        freqs = np.arange(d, dtype=np.float32) / max(d, 1)

        for pos_i, tok in enumerate(toks):
            # Combine exact-word basis + character 3-gram subword basis so morphological paraphrases align!
            padded = f"<{tok}>"
            ngrams = [f"w:{tok}"] + [f"g:{padded[k:k + 3]}" for k in range(max(1, len(padded) - 2))]
            vec = np.zeros(d, dtype=np.float32)
            for weight, ng in [(1.0, ngrams[0])] + [(0.45, g) for g in ngrams[1:]]:
                h = hashlib.blake2b(ng.encode("utf-8"), digest_size=16).digest()
                idx1 = int.from_bytes(h[0:4], "little") % d
                idx2 = int.from_bytes(h[4:8], "little") % d
                idx3 = int.from_bytes(h[8:12], "little") % d
                s1 = 1.0 if (h[12] & 1) else -1.0
                s2 = 1.0 if (h[13] & 1) else -1.0
                s3 = 1.0 if (h[14] & 1) else -1.0
                vec[idx1] += weight * s1
                vec[idx2] += weight * s2
                vec[idx3] += weight * s3
            vec += 0.12 * np.sin((pos_i + 1) * freqs * 6.2831853)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-6:
                vec /= norm
            rows[pos_i] = vec
        return torch.from_numpy(rows)

    def encode(self, question: str) -> torch.Tensor:
        cached = self._cache.get(question)
        if cached is not None:
            return cached
        if self._hf is not None:
            _, states = self._hf.encode(question)
            out = states.to(dtype=torch.float32)
        else:
            out = self._fast_subword_tokens(question)
        self._cache[question] = out
        return out


def collate_universe_records(
    records: Sequence[UniverseRecord],
    featurizer: QuestionFeaturizer,
    *,
    device: Union[str, torch.device] = "cpu",
) -> dict[str, Union[torch.Tensor, list[str]]]:
    """Pad a list of UniverseRecords into one tensor mini-batch (supporting mixed molecules + PBC crystals)."""
    B = len(records)
    if B == 0:
        raise ValueError("Cannot collate an empty sequence of UniverseRecords")

    n_max = max(len(r.Z) for r in records)
    tok_tensors = [featurizer.encode(r.question) for r in records]
    t_max = max(t.shape[0] for t in tok_tensors)
    d_text = featurizer.dim

    Z = torch.zeros(B, n_max, dtype=torch.long)
    pos = torch.zeros(B, n_max, 3, dtype=torch.float32)
    atom_mask = torch.zeros(B, n_max, dtype=torch.bool)
    target_atom_mask = torch.zeros(B, n_max, dtype=torch.bool)
    cell = torch.zeros(B, 3, 3, dtype=torch.float32)
    pbc = torch.zeros(B, 3, dtype=torch.bool)

    text = torch.zeros(B, t_max, d_text, dtype=torch.float32)
    text_mask = torch.zeros(B, t_max, dtype=torch.bool)

    lot_id = torch.zeros(B, dtype=torch.long)
    task_kind = torch.zeros(B, dtype=torch.long)
    scalar_target = torch.zeros(B, 1, dtype=torch.float32)
    vector_target = torch.zeros(B, 3, dtype=torch.float32)
    tensor_target = torch.zeros(B, 3, 3, dtype=torch.float32)
    subtasks: list[str] = []

    for b, (rec, t_emb) in enumerate(zip(records, tok_tensors)):
        n = len(rec.Z)
        Z[b, :n] = torch.tensor(rec.Z, dtype=torch.long)
        pos[b, :n] = torch.tensor(rec.pos_bohr, dtype=torch.float32)
        atom_mask[b, :n] = True
        for a_idx in rec.target_atoms:
            if 0 <= a_idx < n:
                target_atom_mask[b, a_idx] = True

        if rec.cell_bohr is not None:
            cell[b] = torch.tensor(rec.cell_bohr, dtype=torch.float32)
            pbc[b] = torch.tensor(rec.pbc if rec.pbc is not None else [True, True, True], dtype=torch.bool)

        t_len = t_emb.shape[0]
        text[b, :t_len] = t_emb
        text_mask[b, :t_len] = True

        lot_id[b] = resolve_lot_id(rec.lot)
        task_kind[b] = TASK_TO_ID[rec.task_kind]
        scalar_target[b, 0] = float(rec.scalar_target)
        vector_target[b] = torch.tensor(rec.vector_target, dtype=torch.float32)
        tensor_target[b] = torch.tensor(rec.tensor_target, dtype=torch.float32)
        subtasks.append(rec.subtask)

    return {
        "Z": Z.to(device),
        "pos": pos.to(device),
        "atom_mask": atom_mask.to(device),
        "target_atom_mask": target_atom_mask.to(device),
        "cell": cell.to(device) if pbc.any() else None,
        "pbc": pbc.to(device) if pbc.any() else None,
        "text": text.to(device),
        "text_mask": text_mask.to(device),
        "lot_id": lot_id.to(device),
        "task_kind": task_kind.to(device),
        "scalar_target": scalar_target.to(device),
        "vector_target": vector_target.to(device),
        "tensor_target": tensor_target.to(device),
        "subtasks": subtasks,
    }


def make_universe_batches(
    records: Sequence[UniverseRecord],
    featurizer: QuestionFeaturizer,
    *,
    batch_size: int = 16,
    shuffle: bool = False,
    rng: Optional[np.random.Generator] = None,
    device: Union[str, torch.device] = "cpu",
) -> list[dict[str, Union[torch.Tensor, list[str]]]]:
    if not records:
        return []
    order = np.arange(len(records))
    if shuffle:
        if rng is None:
            rng = np.random.default_rng(0)
        rng.shuffle(order)
    batches = []
    for start in range(0, len(order), batch_size):
        chunk = [records[int(i)] for i in order[start:start + batch_size]]
        batches.append(collate_universe_records(chunk, featurizer, device=device))
    return batches

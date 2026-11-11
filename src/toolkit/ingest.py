"""Universal 3D structure, electrolyte formulation, crystal, and MD trajectory ingestor."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import re
from typing import Iterable, Optional, Sequence, Union

import numpy as np

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
BOHR_TO_ANGSTROM = 0.529177210903
AU_DIPOLE_TO_DEBYE = 2.541746473
AU_FORCE_TO_EV_ANG = 51.4220674763

ELEMENTS = (
    "",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
SYMBOL_TO_Z = {sym: idx for idx, sym in enumerate(ELEMENTS) if sym}
SYMBOL_TO_Z.update({sym.upper(): idx for idx, sym in enumerate(ELEMENTS) if sym})

# Approximate covalent radii in Angstrom for bond / clash / proton-transfer heuristics.
COVALENT_RADII_ANG = {
    1: 0.31, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
    19: 2.03, 20: 1.76, 26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22,
    35: 1.20, 53: 1.39,
}

# Built-in 3D conformers (Angstrom) for common demo SMILES so offline environments without RDKit still work.
_BUILTIN_SMILES_ANGSTROM = {
    "O": ([8, 1, 1], [[0.000, 0.000, 0.117], [0.000, 0.757, -0.469], [0.000, -0.757, -0.469]]),
    "H2O": ([8, 1, 1], [[0.000, 0.000, 0.117], [0.000, 0.757, -0.469], [0.000, -0.757, -0.469]]),
    "O=C=O": ([6, 8, 8], [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]),
    "CCO": (
        [6, 6, 8, 1, 1, 1, 1, 1, 1],
        [
            [-1.18, -0.35, 0.00], [0.05, 0.53, 0.00], [1.21, -0.28, 0.00],
            [-2.09, 0.25, 0.00], [-1.19, -0.99, 0.89], [-1.19, -0.99, -0.89],
            [0.05, 1.17, 0.89], [0.05, 1.17, -0.89], [1.98, 0.29, 0.00],
        ],
    ),
    "[Li+]": ([3], [[0.0, 0.0, 0.0]]),
    "O=C1OCCO1": (  # Ethylene carbonate (EC)
        [8, 6, 8, 6, 6, 8, 1, 1, 1, 1],
        [
            [0.00, 2.24, 0.00], [0.00, 1.04, 0.00], [1.12, 0.26, 0.00],
            [0.77, -1.13, 0.12], [-0.77, -1.13, -0.12], [-1.12, 0.26, 0.00],
            [1.15, -1.68, -0.74], [1.19, -1.55, 1.03], [-1.15, -1.68, 0.74], [-1.19, -1.55, -1.03],
        ],
    ),
}


def symbol_for_z(z: int) -> str:
    z = int(z)
    return ELEMENTS[z] if 1 <= z < len(ELEMENTS) else f"X{z}"


def z_for_symbol(token: str) -> int:
    clean = re.sub(r"[^A-Za-z]", "", token.strip())
    if not clean:
        if token.strip().isdigit():
            return int(token.strip())
        raise ValueError(f"Cannot parse element symbol {token!r}")
    norm = clean[0].upper() + clean[1:].lower()
    if norm in SYMBOL_TO_Z:
        return SYMBOL_TO_Z[norm]
    raise ValueError(f"Unknown element symbol {token!r}")


def covalent_radius(z: int) -> float:
    return COVALENT_RADII_ANG.get(int(z), 1.40)


def minimum_image_delta_bohr(
    pos_curr: np.ndarray,
    pos_prev: np.ndarray,
    cell_bohr: Optional[np.ndarray] = None,
    pbc: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute (pos_curr - pos_prev) in bohr, wrapping periodic axes via minimum-image convention."""
    delta = np.asarray(pos_curr, dtype=np.float64) - np.asarray(pos_prev, dtype=np.float64)
    if cell_bohr is None:
        return delta
    cell = np.asarray(cell_bohr, dtype=np.float64)
    pbc_mask = np.ones(3, dtype=bool) if pbc is None else np.asarray(pbc, dtype=bool)
    if not pbc_mask.any():
        return delta
    basis = cell[pbc_mask]
    if np.linalg.matrix_rank(basis) < basis.shape[0]:
        return delta
    pinv = np.linalg.pinv(basis)
    frac = delta @ pinv
    return delta - np.round(frac) @ basis


@dataclass
class StructureFrame:
    """Normalized 3D chemical structure or trajectory frame."""

    Z: np.ndarray
    pos_bohr: np.ndarray
    cell_bohr: Optional[np.ndarray] = None
    pbc: Optional[np.ndarray] = None
    role: Optional[np.ndarray] = None
    amount: Optional[np.ndarray] = None
    species: Optional[np.ndarray] = None
    delta_pos_bohr: Optional[np.ndarray] = None
    forces_ev_ang: Optional[np.ndarray] = None
    dipole_debye: Optional[np.ndarray] = None
    label: str = "structure"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.Z = np.asarray(self.Z, dtype=np.int64).reshape(-1)
        n = len(self.Z)
        self.pos_bohr = np.asarray(self.pos_bohr, dtype=np.float64).reshape(n, 3)
        if self.cell_bohr is not None:
            self.cell_bohr = np.asarray(self.cell_bohr, dtype=np.float64).reshape(3, 3)
            if self.pbc is None:
                self.pbc = np.linalg.norm(self.cell_bohr, axis=-1) > 0
            else:
                self.pbc = np.asarray(self.pbc, dtype=bool).reshape(3)
        elif self.pbc is not None:
            self.pbc = np.asarray(self.pbc, dtype=bool).reshape(3)
        self.role = (np.zeros(n, dtype=np.int64) if self.role is None
                     else np.asarray(self.role, dtype=np.int64).reshape(n))
        self.amount = (np.ones(n, dtype=np.float64) if self.amount is None
                       else np.asarray(self.amount, dtype=np.float64).reshape(n))
        self.species = (np.zeros(n, dtype=np.int64) if self.species is None
                        else np.asarray(self.species, dtype=np.int64).reshape(n))
        if self.delta_pos_bohr is not None:
            self.delta_pos_bohr = np.asarray(self.delta_pos_bohr, dtype=np.float64).reshape(n, 3)
        if self.forces_ev_ang is not None:
            self.forces_ev_ang = np.asarray(self.forces_ev_ang, dtype=np.float64).reshape(n, 3)
        if self.dipole_debye is not None:
            self.dipole_debye = np.asarray(self.dipole_debye, dtype=np.float64).reshape(3)

    @property
    def n_atoms(self) -> int:
        return int(len(self.Z))

    @property
    def symbols(self) -> list[str]:
        return [symbol_for_z(z) for z in self.Z]

    @property
    def pos_angstrom(self) -> np.ndarray:
        return self.pos_bohr * BOHR_TO_ANGSTROM

    @property
    def cell_angstrom(self) -> Optional[np.ndarray]:
        return None if self.cell_bohr is None else self.cell_bohr * BOHR_TO_ANGSTROM

    @property
    def delta_pos_angstrom(self) -> Optional[np.ndarray]:
        return None if self.delta_pos_bohr is None else self.delta_pos_bohr * BOHR_TO_ANGSTROM

    def to_geom_dict(self) -> dict:
        geom = {
            "Z": self.Z.copy(),
            "pos": self.pos_bohr.copy(),
            "role": self.role.copy(),
            "amount": self.amount.copy(),
            "species": self.species.copy(),
        }
        if self.cell_bohr is not None:
            geom["cell"] = self.cell_bohr.copy()
            geom["pbc"] = (self.pbc.copy() if self.pbc is not None else np.ones(3, dtype=bool))
        if self.delta_pos_bohr is not None:
            geom["delta_pos"] = self.delta_pos_bohr.copy()
        return geom

    def transformed(self, R: np.ndarray, translation_bohr: Optional[np.ndarray] = None) -> "StructureFrame":
        """Apply an O(3) matrix R (x -> x @ R.T) and optional translation (bohr) consistently to all 1o fields."""
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t = np.zeros(3, dtype=np.float64) if translation_bohr is None else np.asarray(translation_bohr, dtype=np.float64)
        return StructureFrame(
            Z=self.Z.copy(),
            pos_bohr=self.pos_bohr @ R.T + t[None, :],
            cell_bohr=None if self.cell_bohr is None else self.cell_bohr @ R.T,
            pbc=None if self.pbc is None else self.pbc.copy(),
            role=self.role.copy(),
            amount=self.amount.copy(),
            species=self.species.copy(),
            delta_pos_bohr=None if self.delta_pos_bohr is None else self.delta_pos_bohr @ R.T,
            forces_ev_ang=None if self.forces_ev_ang is None else self.forces_ev_ang @ R.T,
            dipole_debye=None if self.dipole_debye is None else self.dipole_debye @ R.T,
            label=self.label,
            metadata=dict(self.metadata),
        )

    def to_xyz_string(self) -> str:
        header_parts = [f"label={self.label}"]
        if self.cell_angstrom is not None:
            flat = " ".join(f"{v:.6f}" for v in self.cell_angstrom.reshape(-1))
            header_parts.append(f'Lattice="{flat}"')
        lines = [str(self.n_atoms), " ".join(header_parts)]
        pos_ang = self.pos_angstrom
        for z, (x, y, zc) in zip(self.Z, pos_ang):
            lines.append(f"{symbol_for_z(int(z)):2s} {x:12.6f} {y:12.6f} {zc:12.6f}")
        return "\n".join(lines) + "\n"


def _parse_extxyz_header(comment: str) -> dict:
    """Parse key=value or key=\"...\" pairs in an extended XYZ comment line."""
    info = {}
    for m in re.finditer(r'([A-Za-z0-9_]+)=(?:"([^"]*)"|\'([^\']*)\'|(\S+))', comment):
        key = m.group(1)
        val = m.group(2) if m.group(2) is not None else (m.group(3) if m.group(3) is not None else m.group(4))
        info[key] = val
    return info


def parse_xyz_frames(text: str, default_label: str = "xyz") -> list[StructureFrame]:
    """Parse single-frame or multi-frame .xyz / .extxyz text (including Lattice, Properties, role, forces, vel)."""
    raw_lines = [ln.rstrip("\r") for ln in text.splitlines()]
    idx = 0
    frames: list[StructureFrame] = []
    while idx < len(raw_lines):
        while idx < len(raw_lines) and not raw_lines[idx].strip():
            idx += 1
        if idx >= len(raw_lines):
            break
        n_atoms = int(raw_lines[idx].strip())
        if idx + 1 + n_atoms >= len(raw_lines) + 1:
            raise ValueError(f"Truncated XYZ frame starting at line {idx + 1}")
        comment = raw_lines[idx + 1]
        atom_lines = raw_lines[idx + 2: idx + 2 + n_atoms]
        idx += 2 + n_atoms

        info = _parse_extxyz_header(comment)
        cell_bohr = None
        pbc = None
        if "Lattice" in info:
            nums = [float(x) for x in info["Lattice"].split()]
            if len(nums) == 9:
                cell_bohr = np.array(nums, dtype=np.float64).reshape(3, 3) * ANGSTROM_TO_BOHR
                pbc = np.ones(3, dtype=bool)
        if "pbc" in info:
            toks = info["pbc"].split()
            if len(toks) == 3:
                pbc = np.array([t.upper() in ("T", "TRUE", "1") for t in toks], dtype=bool)
        dipole_debye = None
        if "dipole" in info:
            dnums = [float(x) for x in info["dipole"].split()]
            if len(dnums) == 3:
                dipole_debye = np.array(dnums, dtype=np.float64)
        for e_key in ("energy", "total_energy", "mace_energy", "nequip_energy", "dft_energy"):
            if e_key in info:
                try:
                    info.setdefault("energy_ev", float(info[e_key]))
                except ValueError:
                    pass
        for b_key in ("stereo_ok", "relaxed", "converged", "has_energy"):
            if b_key in info and isinstance(info[b_key], str):
                info[b_key] = info[b_key].strip().upper() in ("T", "TRUE", "1", "YES")

        # Parse optional Properties=species:S:1:pos:R:3:forces:R:3:role:I:1...
        prop_spec = info.get("Properties", "species:S:1:pos:R:3")
        sp_parts = prop_spec.split(":")
        columns = []
        col_cursor = 0
        for k in range(0, len(sp_parts) - 2, 3):
            cname, ctype, ccount = sp_parts[k], sp_parts[k + 1], int(sp_parts[k + 2])
            columns.append((cname.lower(), ctype, col_cursor, col_cursor + ccount))
            col_cursor += ccount

        zs, coords, forces, vel, roles, species_ids = [], [], [], [], [], []
        for line in atom_lines:
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed XYZ atom line: {line!r}")
            if columns and len(parts) >= col_cursor:
                row_z = None
                row_pos = None
                for cname, _, c0, c1 in columns:
                    seg = parts[c0:c1]
                    if cname in ("species", "Z", "z"):
                        row_z = int(seg[0]) if seg[0].isdigit() else z_for_symbol(seg[0])
                    elif cname == "pos":
                        row_pos = [float(x) for x in seg]
                    elif cname in ("forces", "force", "mace_forces", "nequip_forces", "dft_forces"):
                        forces.append([float(x) for x in seg])
                    elif cname in ("vel", "velo", "velocity", "delta_pos"):
                        vel.append([float(x) for x in seg])
                    elif cname == "role":
                        roles.append(int(float(seg[0])))
                    elif cname in ("mol_id", "species_id"):
                        species_ids.append(int(float(seg[0])))
                zs.append(z_for_symbol(parts[0]) if row_z is None else row_z)
                coords.append([float(x) for x in parts[1:4]] if row_pos is None else row_pos)
            else:
                zs.append(int(parts[0]) if parts[0].isdigit() else z_for_symbol(parts[0]))
                coords.append([float(x) for x in parts[1:4]])
                if len(parts) >= 7:
                    forces.append([float(x) for x in parts[4:7]])

        pos_bohr = np.asarray(coords, dtype=np.float64) * ANGSTROM_TO_BOHR
        delta_bohr = (np.asarray(vel, dtype=np.float64) * ANGSTROM_TO_BOHR
                      if len(vel) == n_atoms else None)
        forces_arr = np.asarray(forces, dtype=np.float64) if len(forces) == n_atoms else None
        role_arr = np.asarray(roles, dtype=np.int64) if len(roles) == n_atoms else None
        sp_arr = np.asarray(species_ids, dtype=np.int64) if len(species_ids) == n_atoms else None
        lbl = info.get("label") or (comment.strip()[:48] if comment.strip() and "=" not in comment else f"{default_label}_{len(frames)}")
        frames.append(StructureFrame(
            Z=zs, pos_bohr=pos_bohr, cell_bohr=cell_bohr, pbc=pbc,
            role=role_arr, species=sp_arr, delta_pos_bohr=delta_bohr,
            forces_ev_ang=forces_arr, dipole_debye=dipole_debye,
            label=lbl, metadata=info,
        ))

    # Populate inter-frame minimum-image displacements when frames lack explicit velocities/displacements.
    for t in range(1, len(frames)):
        if frames[t].delta_pos_bohr is None and frames[t].n_atoms == frames[t - 1].n_atoms:
            if np.array_equal(frames[t].Z, frames[t - 1].Z):
                frames[t].delta_pos_bohr = minimum_image_delta_bohr(
                    frames[t].pos_bohr, frames[t - 1].pos_bohr, frames[t].cell_bohr, frames[t].pbc
                )
    if len(frames) > 1 and frames[0].delta_pos_bohr is None:
        frames[0].delta_pos_bohr = np.zeros_like(frames[0].pos_bohr)
    return frames


def parse_poscar(text: str, label: str = "poscar") -> StructureFrame:
    """Lightweight VASP POSCAR / CONTCAR parser (Direct or Cartesian coordinates)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 8:
        raise ValueError("Truncated POSCAR content")
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=np.float64) * scale
    elem_tokens = lines[5].split()
    if elem_tokens[0].isdigit():
        raise ValueError("VASP4 POSCAR without element symbol line is unsupported; include element symbols on line 6")
    counts = [int(x) for x in lines[6].split()]
    cursor = 7
    if lines[cursor][0].lower() == "s":  # Selective dynamics
        cursor += 1
    coord_mode = lines[cursor][0].lower()
    cursor += 1
    zs = []
    for sym, cnt in zip(elem_tokens, counts):
        zs.extend([z_for_symbol(sym)] * cnt)
    total = len(zs)
    raw = np.array([[float(x) for x in lines[cursor + i].split()[:3]] for i in range(total)], dtype=np.float64)
    pos_ang = (raw % 1.0) @ lattice if coord_mode in ("d", "f") else raw * (scale if scale > 0 else 1.0)
    return StructureFrame(
        Z=zs,
        pos_bohr=pos_ang * ANGSTROM_TO_BOHR,
        cell_bohr=lattice * ANGSTROM_TO_BOHR,
        pbc=np.ones(3, dtype=bool),
        label=label,
    )


def parse_sdf_v2000(text: str, label: str = "sdf") -> StructureFrame:
    """Parse V2000 MOL/SDF block without requiring RDKit."""
    lines = text.splitlines()
    counts_idx = next((i for i, ln in enumerate(lines[:10]) if "V2000" in ln), 3)
    n_atoms = int(lines[counts_idx][:3].strip())
    zs, coords = [], []
    for ln in lines[counts_idx + 1: counts_idx + 1 + n_atoms]:
        parts = ln.split()
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
        zs.append(z_for_symbol(parts[3]))
    return StructureFrame(Z=zs, pos_bohr=np.asarray(coords, dtype=np.float64) * ANGSTROM_TO_BOHR, label=label)


def parse_simple_cif(text: str, label: str = "cif") -> StructureFrame:
    """Parse basic P1 / explicit-site CIF files offline; delegates to pymatgen when available."""
    try:
        from pymatgen.core import Structure
        s = Structure.from_str(text, fmt="cif")
        lattice = np.asarray(s.lattice.matrix, dtype=np.float64)
        zs = [int(site.specie.Z) for site in s]
        pos_ang = (np.asarray([site.frac_coords for site in s], dtype=np.float64) % 1.0) @ lattice
        return StructureFrame(
            Z=zs, pos_bohr=pos_ang * ANGSTROM_TO_BOHR,
            cell_bohr=lattice * ANGSTROM_TO_BOHR, pbc=np.ones(3, dtype=bool), label=label,
        )
    except ImportError:
        pass

    def _num(pattern: str, default: float) -> float:
        m = re.search(pattern + r"\s+([0-9.+-]+)", text)
        return float(m.group(1)) if m else default

    a = _num(r"_cell_length_a", 5.0)
    b = _num(r"_cell_length_b", a)
    c = _num(r"_cell_length_c", a)
    alpha = math.radians(_num(r"_cell_angle_alpha", 90.0))
    beta = math.radians(_num(r"_cell_angle_beta", 90.0))
    gamma = math.radians(_num(r"_cell_angle_gamma", 90.0))
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * math.cos(gamma), b * math.sin(gamma), 0.0])
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / max(math.sin(gamma), 1e-8)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 1e-12))
    lattice = np.vstack([va, vb, np.array([cx, cy, cz])])

    zs, fracs = [], []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith(("#", "_", "loop_", "data_")):
            continue
        parts = s.split()
        if len(parts) >= 4:
            try:
                sym_z = z_for_symbol(parts[0])
                fx, fy, fz = (float(re.sub(r"\([^)]*\)", "", parts[k])) for k in (-3, -2, -1))
                zs.append(sym_z)
                fracs.append([fx % 1.0, fy % 1.0, fz % 1.0])
            except ValueError:
                continue
    if not zs:
        raise ValueError("Could not parse atomic sites from CIF (install pymatgen for space-group symmetry expansion)")
    pos_ang = np.asarray(fracs, dtype=np.float64) @ lattice
    return StructureFrame(
        Z=zs, pos_bohr=pos_ang * ANGSTROM_TO_BOHR,
        cell_bohr=lattice * ANGSTROM_TO_BOHR, pbc=np.ones(3, dtype=bool), label=label,
    )


def smiles_to_frame(smiles: str, seed: int = 0, role: int = 0, amount: float = 1.0) -> StructureFrame:
    """Convert a SMILES string into a 3D StructureFrame (RDKit ETKDG+MMFF94, with built-in demo fallback)."""
    s = smiles.strip()
    if s in _BUILTIN_SMILES_ANGSTROM:
        try:
            from src.tasks.conformers import smiles_to_conformer
            z_arr, pos_bohr = smiles_to_conformer(s, seed=seed)
            return StructureFrame(
                Z=z_arr, pos_bohr=pos_bohr,
                role=np.full(len(z_arr), role, dtype=np.int64),
                amount=np.full(len(z_arr), amount, dtype=np.float64),
                label=s, metadata={"smiles": s},
            )
        except Exception:
            z_list, pos_ang = _BUILTIN_SMILES_ANGSTROM[s]
            pos = np.asarray(pos_ang, dtype=np.float64)
            pos = (pos - pos.mean(0)) * ANGSTROM_TO_BOHR
            return StructureFrame(
                Z=z_list, pos_bohr=pos,
                role=np.full(len(z_list), role, dtype=np.int64),
                amount=np.full(len(z_list), amount, dtype=np.float64),
                label=s, metadata={"smiles": s},
            )
    from src.tasks.conformers import smiles_to_conformer
    z_arr, pos_bohr = smiles_to_conformer(s, seed=seed)
    return StructureFrame(
        Z=z_arr, pos_bohr=pos_bohr,
        role=np.full(len(z_arr), role, dtype=np.int64),
        amount=np.full(len(z_arr), amount, dtype=np.float64),
        label=s, metadata={"smiles": s},
    )


def build_solvation_cluster(
    solvent_smiles: Sequence[str] = ("O=C1OCCO1", "O=C1OCCO1", "O=C1OCCO1"),
    ion_symbol: str = "Li",
    coord_distance_angstrom: float = 2.05,
    seed: int = 0,
) -> StructureFrame:
    """Assemble a 3D first-solvation-shell cluster around a central working ion with coordinating atoms oriented inward."""
    ion_z = z_for_symbol(ion_symbol)
    n_lig = len(solvent_smiles)
    zs = [ion_z]
    coords_ang = [np.zeros(3, dtype=np.float64)]
    roles = [1]       # salt / working ion
    amounts = [1.0]
    species_ids = [0]

    # Golden-spiral directions on S^2 for well-separated coordination shell placement.
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for idx, smi in enumerate(solvent_smiles):
        lig = smiles_to_frame(smi, seed=seed + idx, role=2)
        lig_ang = lig.pos_angstrom.copy()
        # Pick the most electronegative donor atom (O, N, F, S) to point toward the central ion.
        donor_candidates = [i for i, z in enumerate(lig.Z) if z in (8, 7, 9, 16, 17)]
        donor_idx = donor_candidates[0] if donor_candidates else 0
        centroid = lig_ang.mean(axis=0)
        donor_vec = lig_ang[donor_idx] - centroid
        norm_dv = float(np.linalg.norm(donor_vec))

        y = 1.0 - (2.0 * (idx + 0.5) / max(n_lig, 1))
        radius_xy = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * idx
        target_dir = np.array([math.cos(theta) * radius_xy, y, math.sin(theta) * radius_xy], dtype=np.float64)
        # Rotate ligand so donor_vec points along -target_dir (donor faces ion at origin).
        if norm_dv > 1e-6:
            u = donor_vec / norm_dv
            v = -target_dir
            c = float(np.clip(np.dot(u, v), -1.0, 1.0))
            if c < -0.9999:
                perp = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                axis = np.cross(u, perp)
                axis /= np.linalg.norm(axis)
                R = -np.eye(3) + 2.0 * np.outer(axis, axis)
            elif c < 0.9999:
                vx = np.cross(u, v)
                s = float(np.linalg.norm(vx))
                K = np.array([[0, -vx[2], vx[1]], [vx[2], 0, -vx[0]], [-vx[1], vx[0], 0]], dtype=np.float64)
                R = np.eye(3) + K + (K @ K) * ((1.0 - c) / (s * s))
            else:
                R = np.eye(3)
            lig_centered = (lig_ang - centroid) @ R.T
        else:
            lig_centered = lig_ang - centroid

        shift = target_dir * coord_distance_angstrom - lig_centered[donor_idx]
        placed = lig_centered + shift[None, :]
        zs.extend(lig.Z.tolist())
        coords_ang.extend(placed)
        roles.extend([2] * lig.n_atoms)
        amounts.extend([1.0 / max(n_lig, 1)] * lig.n_atoms)
        species_ids.extend([idx + 1] * lig.n_atoms)

    return StructureFrame(
        Z=zs,
        pos_bohr=np.asarray(coords_ang, dtype=np.float64) * ANGSTROM_TO_BOHR,
        role=roles,
        amount=amounts,
        species=species_ids,
        label=f"{ion_symbol}+_solvation_shell",
        metadata={"working_ion": ion_symbol, "solvents": list(solvent_smiles)},
    )


def load_trajectory(source: Union[str, Path, Sequence[StructureFrame]]) -> list[StructureFrame]:
    """Load a multi-frame trajectory from .xyz/.extxyz, ASE .traj, or a sequence of StructureFrames."""
    if isinstance(source, (list, tuple)) and all(isinstance(f, StructureFrame) for f in source):
        return list(source)
    path = Path(source)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix in (".xyz", ".extxyz"):
            return parse_xyz_frames(path.read_text(encoding="utf-8"), default_label=path.stem)
        if suffix == ".traj":
            from ase.io import read as ase_read
            atoms_list = ase_read(str(path), index=":")
            frames = [_from_ase_atoms(a, label=f"{path.stem}_{i}") for i, a in enumerate(atoms_list)]
            for t in range(1, len(frames)):
                if frames[t].n_atoms == frames[t - 1].n_atoms:
                    frames[t].delta_pos_bohr = minimum_image_delta_bohr(
                        frames[t].pos_bohr, frames[t - 1].pos_bohr, frames[t].cell_bohr, frames[t].pbc
                    )
            if frames and frames[0].delta_pos_bohr is None:
                frames[0].delta_pos_bohr = np.zeros_like(frames[0].pos_bohr)
            return frames
    return [load_structure(source)]


def _from_ase_atoms(atoms, label: str = "ase") -> StructureFrame:
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    cell_ang = np.asarray(atoms.get_cell().array, dtype=np.float64) if pbc.any() else None
    forces = None
    try:
        if atoms.calc is not None and "forces" in atoms.calc.results:
            forces = np.asarray(atoms.calc.results["forces"], dtype=np.float64)
    except Exception:
        forces = None
    if forces is None and hasattr(atoms, "arrays"):
        for f_key in ("forces", "mace_forces", "nequip_forces", "dft_forces"):
            if f_key in atoms.arrays:
                forces = np.asarray(atoms.arrays[f_key], dtype=np.float64)
                break
    meta = dict(getattr(atoms, "info", {}) or {})
    for e_key in ("energy", "total_energy", "mace_energy", "nequip_energy"):
        if e_key in meta:
            try:
                meta.setdefault("energy_ev", float(meta[e_key]))
            except (TypeError, ValueError):
                pass
    return StructureFrame(
        Z=atoms.get_atomic_numbers(),
        pos_bohr=np.asarray(atoms.get_positions(), dtype=np.float64) * ANGSTROM_TO_BOHR,
        cell_bohr=None if cell_ang is None else cell_ang * ANGSTROM_TO_BOHR,
        pbc=pbc if pbc.any() else None,
        forces_ev_ang=forces,
        label=str(meta.get("source_id") or meta.get("label") or label),
        metadata=meta,
    )


def load_structure(source: Union[str, Path, dict, StructureFrame], frame_index: int = -1) -> StructureFrame:
    """Universal single-structure loader: StructureFrame, geom dict, file path (.xyz/.extxyz/.sdf/.mol/POSCAR/.cif), or SMILES."""
    if isinstance(source, StructureFrame):
        return source
    if isinstance(source, dict) and "Z" in source and "pos" in source:
        return StructureFrame(
            Z=source["Z"],
            pos_bohr=source["pos"],
            cell_bohr=source.get("cell"),
            pbc=source.get("pbc"),
            role=source.get("role"),
            amount=source.get("amount"),
            species=source.get("species"),
            delta_pos_bohr=source.get("delta_pos"),
            label=str(source.get("label", "dict_geom")),
        )
    if hasattr(source, "get_atomic_numbers") and hasattr(source, "get_positions"):
        return _from_ase_atoms(source)

    s = str(source)
    p = Path(s)
    if "\n" not in s and len(s) < 260 and p.exists() and p.is_file():
        name_lower = p.name.lower()
        suffix = p.suffix.lower()
        content = p.read_text(encoding="utf-8")
        if suffix in (".xyz", ".extxyz"):
            frames = parse_xyz_frames(content, default_label=p.stem)
            return frames[frame_index]
        if suffix in (".sdf", ".mol"):
            return parse_sdf_v2000(content, label=p.stem)
        if suffix == ".cif":
            return parse_simple_cif(content, label=p.stem)
        if suffix in (".vasp", ".poscar") or name_lower.startswith(("poscar", "contcar")):
            return parse_poscar(content, label=p.stem)
        if suffix == ".traj":
            return load_trajectory(p)[frame_index]

    # Raw inline multi-line XYZ text
    if "\n" in s:
        first = s.strip().splitlines()[0].strip()
        if first.isdigit():
            return parse_xyz_frames(s)[frame_index]
        if "V2000" in s:
            return parse_sdf_v2000(s)

    # Fallback: treat string as SMILES
    return smiles_to_frame(s)

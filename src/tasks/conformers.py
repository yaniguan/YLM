"""3D structures for SMILES (RDKit ETKDGv3 + MMFF94) and mixture assembly.

* deterministic: fixed ETKDG random seed, one thread.
* multi-fragment SMILES (salts, e.g. "[Li+].F[P-](F)(F)(F)(F)F"): every fragment is embedded on its own
  and fragments are placed more than one encoder cutoff apart, i.e. as dissociated ions. This is an
  approximation (no contact ion pairs); it keeps the encoder graph of each ion well defined.
* MMFF94 when all atoms are parameterised, else UFF, else the raw ETKDG geometry.
* results are cached in data/cache/conformers.pkl, keyed by (smiles, seed).
Positions are returned in bohr, centred on the centroid.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
BOHR_TO_ANGSTROM = 0.529177210903
ENCODER_CUTOFF_BOHR = 9.45
DEFAULT_CACHE = "data/cache/conformers.pkl"


class ConformerError(RuntimeError):
    pass


def place_apart(blocks: list, min_gap: float) -> list:
    """Translate point clouds (each (n, 3)) along x so atoms of different blocks are > min_gap apart."""
    out, cursor = [], 0.0
    for pos in blocks:
        pos = np.asarray(pos, dtype=np.float64)
        centred = pos - pos.mean(0)
        r = float(np.linalg.norm(centred, axis=1).max()) if len(pos) else 0.0
        centre = cursor + r
        out.append(centred + np.array([centre, 0.0, 0.0]))
        cursor = centre + r + min_gap
    return out


def _embed_fragment(frag, seed: int):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(frag)
    if mol.GetNumAtoms() == 1:
        return np.array([a.GetAtomicNum() for a in mol.GetAtoms()]), np.zeros((1, 3))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ConformerError(f"ETKDG failed for {Chem.MolToSmiles(frag)}")
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        elif AllChem.UFFHasAllMoleculeParams(mol):
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        pass  # keep the ETKDG geometry
    Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    pos = mol.GetConformer().GetPositions()
    return Z, pos


def smiles_to_conformer(smiles: str, seed: int = 0) -> tuple:
    """-> (Z (N,) int64, pos (N, 3) bohr). Raises ConformerError."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ConformerError(f"RDKit cannot parse {smiles!r}")
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    Zs, blocks = [], []
    for frag in frags:
        Z, pos = _embed_fragment(frag, seed)
        Zs.append(Z)
        blocks.append(pos * ANGSTROM_TO_BOHR)
    blocks = place_apart(blocks, ENCODER_CUTOFF_BOHR + 2.0) if len(blocks) > 1 else blocks
    pos = np.concatenate(blocks)
    return np.concatenate(Zs), pos - pos.mean(0)


class ConformerCache:
    def __init__(self, path: Optional[str] = DEFAULT_CACHE):
        self.path = path
        self.data = {}
        self.dirty = False
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                self.data = pickle.load(f)

    def get(self, smiles: str, seed: int = 0):
        """-> (Z, pos_bohr) or None if embedding failed (failures are cached too)."""
        key = f"{smiles}|{seed}"
        if key not in self.data:
            self.data[key] = _safe_conformer(smiles, seed)
            self.dirty = True
        return self.data[key]

    def fill(self, smiles_list, seed: int = 0, n_jobs: int = 1):
        todo = sorted({s for s in smiles_list if f"{s}|{seed}" not in self.data})
        if not todo:
            return
        if n_jobs > 1:
            from multiprocessing import Pool
            with Pool(n_jobs) as pool:
                results = pool.starmap(_safe_conformer, [(s, seed) for s in todo], chunksize=16)
        else:
            results = [_safe_conformer(s, seed) for s in todo]
        for s, r in zip(todo, results):
            self.data[f"{s}|{seed}"] = r
        self.dirty = True

    def save(self):
        if not (self.path and self.dirty):
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.data, f)
        os.replace(tmp, self.path)
        self.dirty = False


def _safe_conformer(smiles: str, seed: int):
    try:
        return smiles_to_conformer(smiles, seed)
    except ConformerError:
        return None
    except Exception:  # RDKit can raise odd errors on exotic species; treat as failure
        return None


def build_mixture(species: list, min_gap: float = ENCODER_CUTOFF_BOHR + 2.0) -> dict:
    """species: list of dicts {Z, pos (bohr), role (int), amount (float)}.

    Returns one geometry dict {Z, pos, role, amount, species} where species are separated by more than
    the encoder cutoff, so the encoder sees each species as its own graph."""
    blocks = place_apart([s["pos"] for s in species], min_gap)
    Z = np.concatenate([np.asarray(s["Z"], dtype=np.int64) for s in species])
    pos = np.concatenate(blocks)
    role = np.concatenate([np.full(len(s["Z"]), s["role"], dtype=np.int64) for s in species])
    amount = np.concatenate([np.full(len(s["Z"]), s["amount"], dtype=np.float64) for s in species])
    sid = np.concatenate([np.full(len(s["Z"]), i, dtype=np.int64) for i, s in enumerate(species)])
    return {"Z": Z, "pos": pos - pos.mean(0), "role": role, "amount": amount, "species": sid}


def single_geometry(Z, pos_bohr, role: int = 0, amount: float = 1.0) -> dict:
    Z = np.asarray(Z, dtype=np.int64)
    return {"Z": Z, "pos": np.asarray(pos_bohr, dtype=np.float64), "role": np.full(len(Z), role),
            "amount": np.full(len(Z), amount, dtype=np.float64), "species": np.zeros(len(Z), dtype=np.int64)}

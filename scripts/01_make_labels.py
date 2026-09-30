"""Make labels for QM9 molecules with GFN2-xTB.

For each molecule:
  1. read the 3D structure from gdb9.sdf
  2. move every atom a little (Gaussian noise, 0.05 Angstrom) so forces are not ~zero
  3. run GFN2-xTB -> energy, dipole vector, forces

Output: data/qm9_xtb.pkl  (a list of dicts, one per molecule)

Units (atomic units, as xTB gives them):
  positions  bohr
  dipole     e*bohr        (vector, 3 numbers)
  forces     hartree/bohr  (one vector per atom)

Run:
  python scripts/01_make_labels.py --n 20000
"""
import argparse
import os
import pickle
import time
from multiprocessing import Pool

os.environ["OMP_NUM_THREADS"] = "1"  # one thread per worker, we parallelize over molecules

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
NOISE_ANGSTROM = 0.05


def read_molecules(sdf_path):
    """Return a list of RDKit molecules with hydrogens and 3D coordinates."""
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    mols = []
    for i, mol in enumerate(supplier):
        if mol is None:
            continue  # RDKit could not read it; skip
        mol.SetProp("qm9_index", str(i))
        mols.append(mol)
    return mols


def run_xtb(job):
    """job = (qm9_index, atomic_numbers, positions_angstrom, seed). Returns a dict or None."""
    import tblite.interface as tb

    qm9_index, numbers, pos_ang, seed = job
    rng = np.random.default_rng(seed)
    pos_ang = pos_ang + rng.normal(0.0, NOISE_ANGSTROM, size=pos_ang.shape)
    pos_bohr = pos_ang * ANGSTROM_TO_BOHR
    try:
        calc = tb.Calculator("GFN2-xTB", numbers, pos_bohr)
        calc.set("verbosity", 0)
        res = calc.singlepoint()
    except Exception:
        return None  # SCF did not converge etc.; skip this molecule
    return {
        "qm9_index": qm9_index,
        "numbers": numbers,
        "pos": pos_bohr,
        "energy": float(res.get("energy")),
        "dipole": np.array(res.get("dipole")),
        "forces": -np.array(res.get("gradient")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", default="data/gdb9.sdf")
    parser.add_argument("--out", default="data/qm9_xtb.pkl")
    parser.add_argument("--n", type=int, default=20000, help="how many molecules")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("reading", args.sdf)
    mols = read_molecules(args.sdf)
    print("molecules read:", len(mols))

    rng = np.random.default_rng(args.seed)
    picked = rng.choice(len(mols), size=min(args.n, len(mols)), replace=False)

    jobs = []
    for k, i in enumerate(picked):
        mol = mols[i]
        numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
        pos = mol.GetConformer().GetPositions()
        jobs.append((int(mol.GetProp("qm9_index")), numbers, pos, args.seed * 1_000_000 + k))

    t0 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(run_xtb, jobs, chunksize=64)
    results = [r for r in results if r is not None]
    print(f"done: {len(results)} ok / {len(jobs)} jobs in {time.time() - t0:.0f} s")

    with open(args.out, "wb") as f:
        pickle.dump(results, f)
    print("saved", args.out)


if __name__ == "__main__":
    main()

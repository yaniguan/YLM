"""l=2 labels for TensorQA: traceless molecular quadrupole from one PySCF HF/6-31G* single point.

The quadrupole is a symmetric traceless 3x3 tensor, i.e. an l=2 object that rotates as R Q R^T.
It needs one SCF per molecule (seconds), unlike polarizability (finite field, minutes).
Origin: centre of nuclear charge, so the label depends only on the geometry (not on the lab origin).
Geometry: the same displaced geometries as the xTB labels (data/qm9_xtb.pkl), positions in bohr.

  python scripts/12_make_quadrupole.py --n 2000 --out data/qm9_quad.pkl [--start 0 --stop 500]  # slices for array jobs
"""
import argparse
import pickle
import time

import numpy as np
from pyscf import gto, scf

ANG = 0.52917721092


def quadrupole(numbers, pos_bohr, basis="6-31g*"):
    mol = gto.M(atom=[(int(z), tuple(np.asarray(p) * ANG)) for z, p in zip(numbers, pos_bohr)],
                basis=basis, verbose=0, unit="Angstrom")
    mf = scf.RHF(mol)
    mf.kernel()
    if not mf.converged:
        return None
    dm = mf.make_rdm1()
    Z = np.asarray(numbers, float)
    pos = np.asarray(pos_bohr, float)
    origin = (Z[:, None] * pos).sum(0) / Z.sum()
    with mol.with_common_orig(origin):  # origin in bohr
        rr = mol.intor("int1e_rr").reshape(3, 3, mol.nao, mol.nao)
    r = pos - origin
    nuc = np.einsum("i,ia,ib->ab", Z, r, r)
    elec = np.einsum("abmn,mn->ab", rr, dm)
    q = nuc - elec
    return q - np.eye(3) * np.trace(q) / 3.0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="data/qm9_xtb.pkl")
    p.add_argument("--out", default="data/qm9_quad.pkl")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=2000)
    args = p.parse_args()
    labels = pickle.load(open(args.labels, "rb"))
    out, t0 = {}, time.time()
    for m in range(args.start, min(args.stop, len(labels))):
        q = quadrupole(labels[m]["numbers"], labels[m]["pos"])
        if q is not None:
            out[m] = q
        if (m - args.start) % 100 == 0:
            print(m, f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump(out, open(args.out, "wb"))
    print("saved", len(out), args.out)

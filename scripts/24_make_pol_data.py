"""TensorQA-pol: full static polarizability tensors (LR-CCSD/d-aug-cc-pVDZ) of the 7211 QM7b molecules.

Data: Materials Cloud 10.24435/materialscloud:2019.0002 (Yang et al., Sci. Data 6, 152 (2019); CC BY 4.0),
file qm7b_coords.xyz (extended xyz, Angstrom; alpha = axx ayy azz axy axz ayz in atomic units).
  curl -L https://archive.materialscloud.org/api/records/7a0t1-vds18/files/qm7b_coords.xyz/content -o data/qm7b/qm7b_coords.xyz

Questions (the molecule has 0e + 2e polarizability; atom pointers pick a direction u = unit(r_B - r_A)):
  pol_iso    scalar  alpha_iso = tr(alpha)/3
  pol_aniso  2e      the traceless part, five 2e coefficients (e3nn CartesianTensor "ij=ji")
  pol_dir    scalar  u^T alpha u
  pol_ind    1o      alpha u            (dipole induced by a unit field along u)
Splits: molecules grouped by canonical connectivity graph, then 80/10/10; `test_heldout` re-words the test questions
in the held-out style families (chatty, inverted). Bonds come from RDKit DetermineConnectivity (distance based), which is
only used to WORD atom descriptions; the model sees positions and elements only.

  python scripts/24_make_pol_data.py    # -> data/qm7b_pol.pkl, data/tensorqa_pol.json
"""
import importlib.util
import json
import os
import pickle
import random
from collections import Counter, defaultdict

import numpy as np
from e3nn.io import CartesianTensor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

RDLogger.DisableLog("rdApp.*")
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mkq", os.path.join(here, "02_make_questions.py"))
mkq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mkq)

ANG = 1.8897261246   # Angstrom -> bohr
Z_OF = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17}
CART = CartesianTensor("ij=ji")


def read_xyz(path):
    out, lines, i = [], open(path).read().split("\n"), 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        head = lines[i + 1]
        atoms = [l.split() for l in lines[i + 2: i + 2 + n]]
        props = {}
        for key in ("ccsd_pol", "b3lyp_pol", "scan0_pol"):
            a = head.index(key + '="') + len(key) + 2
            props[key] = np.array(head[a: head.index('"', a)].split(), float)
        out.append(dict(numbers=[Z_OF[a[0]] for a in atoms], pos=np.array([a[1:4] for a in atoms], float) * ANG, **props))
        i += n + 2
    return out


def connectivity_mol(numbers, pos_bohr):
    """RDKit mol with distance-based bonds (used only to word atom descriptions and by the rule parser)."""
    rd = Chem.RWMol()
    for z in numbers:
        rd.AddAtom(Chem.Atom(int(z)))
    conf = Chem.Conformer(len(numbers))
    for k, p in enumerate(np.asarray(pos_bohr) / ANG):
        conf.SetAtomPosition(k, tuple(float(x) for x in p))
    rd.AddConformer(conf)
    mol = rd.GetMol()
    rdDetermineBonds.DetermineConnectivity(mol)
    Chem.FastFindRings(mol)
    return mol


def tensor(v):
    xx, yy, zz, xy, xz, yz = v
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def main():
    rng = random.Random(0)
    P = json.load(open("data/paraphrases_pol.json"))
    held = set(P["heldout_styles"])
    styles = {t: {"train": {s: v for s, v in d.items() if s not in held and v},
                  "held": {s: v for s, v in d.items() if s in held and v}} for t, d in P["templates"].items()}
    mols = read_xyz("data/qm7b/qm7b_coords.xyz")
    labels, graph, desc = [], {}, {}
    for m, d in enumerate(mols):
        mol = connectivity_mol(d["numbers"], d["pos"])
        graph[m] = Chem.MolToSmiles(mol)
        desc[m] = mkq.unique_atom_descriptions(mol)
        a = tensor(d["ccsd_pol"])
        labels.append(dict(numbers=d["numbers"], pos=d["pos"], alpha=a, alpha_b3lyp=tensor(d["b3lyp_pol"]),
                           alpha_scan0=tensor(d["scan0_pol"])))
    pickle.dump(labels, open("data/qm7b_pol.pkl", "wb"))
    groups = defaultdict(list)
    for m, g in graph.items():
        groups[g].append(m)
    keys = sorted(groups)
    rng.shuffle(keys)
    split, c, n = {}, 0, len(labels)
    for g in keys:
        sp = "train" if c < 0.8 * n else "val" if c < 0.9 * n else "test"
        for m in groups[g]:
            split[m] = sp
        c += len(groups[g])

    def pick(task, kind):
        st = rng.choice(sorted(styles[task][kind]))
        return rng.choice(styles[task][kind][st]), st

    out = []

    def add(m, sp, task, atoms, target, tmpl, st):
        d = desc[m]
        q = tmpl.format(a=d[atoms[0]], b=d[atoms[1]]) if atoms[0] >= 0 else tmpl
        out.append(dict(mol=m, split=sp, task=task, atoms=list(atoms), target=[float(x) for x in target], question=q, style=st))

    for m, lab in enumerate(labels):
        a = lab["alpha"]
        iso = np.trace(a) / 3
        dev = CART.from_cartesian(__import__("torch").tensor(a))[1:].numpy()    # five 2e coefficients (trace dropped)
        sps = [split[m]] + (["test_heldout"] if split[m] == "test" else [])
        for task, tg in [("pol_iso", [iso]), ("pol_aniso", dev)]:
            for sp in sps:
                t, st = pick(task, "held" if sp == "test_heldout" else "train")
                add(m, sp, task, (-1, -1), tg, t, st)
        ids = sorted(desc[m])
        if len(ids) >= 2:
            for _ in range(2):
                i, j = rng.sample(ids, 2)
                u = lab["pos"][j] - lab["pos"][i]
                u /= np.linalg.norm(u)
                for task, tg in [("pol_dir", [u @ a @ u]), ("pol_ind", a @ u)]:
                    for sp in sps:
                        t, st = pick(task, "held" if sp == "test_heldout" else "train")
                        add(m, sp, task, (i, j), tg, t, st)
    json.dump(out, open("data/tensorqa_pol.json", "w"))
    print("molecules", len(labels), "rows", len(out), "unique questions", len({q["question"] for q in out}))
    print(sorted(Counter((q["split"], q["task"]) for q in out).items()))
    iso = np.array([np.trace(l["alpha"]) / 3 for l in labels])
    b3 = np.array([np.trace(l["alpha_b3lyp"]) / 3 for l in labels])
    print(f"alpha_iso: mean {iso.mean():.1f}, std {iso.std():.1f} a.u.; B3LYP-vs-CCSD rel. RMSE {np.sqrt(((iso - b3) ** 2).mean()) / iso.std():.4f}")
    print("max atoms:", max(len(l["numbers"]) for l in labels), "| molecules with an atom unique-describable:", sum(1 for m in desc if len(desc[m]) >= 2))


if __name__ == "__main__":
    main()

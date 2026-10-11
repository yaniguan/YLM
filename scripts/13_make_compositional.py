"""Compositional TensorQA questions (v0.2 draft). Labels come from the existing xTB forces, no new compute.

  T4 diff  "What is the difference between the force on A and the force on B?"   F_A - F_B
           Needs TWO atoms selected and combined with a sign.
  T5 proj  "What is the part of the force on A that points toward B?"            (F_A . u) u, u = unit(r_B - r_A)
           Needs a direction built from two atoms (Option B/C territory); a plain weighted average of atom
           vectors cannot make it.

Atoms A, B are described as in TensorQA-v0 (unique descriptions only), A != B.
Splits follow tensorqa_v0.json (by molecule). Held-out wordings only in test_heldout.
A plain parser needs a new grammar rule per question type; that is the point of the comparison.

  python scripts/13_make_compositional.py            # -> data/tensorqa_comp.json
"""
import argparse
import importlib.util
import json
import os
import pickle
import random
from collections import Counter

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mkq", os.path.join(here, "02_make_questions.py"))
mkq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mkq)

TEMPLATES = {
    "diff": ["What is the difference between the force on {a} and the force on {b}?",
             "Subtract the force on {b} from the force on {a}.",
             "Give the force on {a} minus the force on {b}."],
    "proj": ["What is the part of the force on {a} that points toward {b}?",
             "Give the component of the force on {a} along the direction to {b}.",
             "Project the force on {a} onto the line pointing to {b}."],
}
HELDOUT_IDX = 2  # last template of each task is only used in test_heldout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sdf", default="data/gdb9.sdf")
    p.add_argument("--labels", default="data/qm9_xtb.pkl")
    p.add_argument("--v0", default="data/tensorqa_v0.json")
    p.add_argument("--out", default="data/tensorqa_comp.json")
    p.add_argument("--pairs_per_task", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    labels = pickle.load(open(args.labels, "rb"))
    split_of = {q["mol"]: q["split"] for q in json.load(open(args.v0)) if q["split"] != "test_heldout"}
    supplier = Chem.SDMolSupplier(args.sdf, removeHs=False, sanitize=True)

    out = []
    for m, split in sorted(split_of.items()):
        lab = labels[m]
        desc = mkq.unique_atom_descriptions(supplier[lab["qm9_index"]])
        atoms = sorted(desc)
        if len(atoms) < 2:
            continue
        pos, forces = np.asarray(lab["pos"], float), np.asarray(lab["forces"], float)
        for task, templates in TEMPLATES.items():
            for _ in range(args.pairs_per_task):
                a, b = rng.sample(atoms, 2)
                if task == "diff":
                    target = forces[a] - forces[b]
                else:
                    u = pos[b] - pos[a]
                    u /= np.linalg.norm(u)
                    target = (forces[a] @ u) * u
                if np.abs(target).max() > 2 * mkq.MAX_FORCE:
                    continue
                splits = [(split, rng.choice(templates[:HELDOUT_IDX]))]
                if split == "test":
                    splits.append(("test_heldout", templates[HELDOUT_IDX]))
                for sp, t in splits:
                    out.append(dict(mol=m, split=sp, task=task, atoms=[a, b], target=target.tolist(),
                                    question=t.format(a=desc[a], b=desc[b])))
    json.dump(out, open(args.out, "w"), indent=0)
    print("questions:", len(out))
    print(Counter((q["split"], q["task"]) for q in out))
    for q in out[:4]:
        print(" ", q["task"], "|", q["question"])


if __name__ == "__main__":
    main()

"""Make TensorQA-v0 questions from the xTB labels.

Two question types for month 1:
  T1  "What is the dipole moment of this molecule?"   -> dipole vector (whole molecule)
  T3  "What is the force on the oxygen atom bonded to two carbons?" -> force vector on one atom

An atom is described in words by: element + what it is bonded to (+ ring, only if needed).
We only ask about an atom if that description matches exactly ONE atom in the molecule.

Templates: some are used for train/val/test, some are held out and only used in "test_heldout".
That tests whether the model understands new wordings.

Output: data/tensorqa_v0.json

Run:
  python scripts/02_make_questions.py
"""
import argparse
import json
import pickle
import random
from collections import Counter

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

NAME = {1: "hydrogen", 6: "carbon", 7: "nitrogen", 8: "oxygen", 9: "fluorine"}
NAME_PLURAL = {1: "hydrogens", 6: "carbons", 7: "nitrogens", 8: "oxygens", 9: "fluorines"}
COUNT_WORD = {1: "one", 2: "two", 3: "three", 4: "four"}

DIPOLE_TEMPLATES = [
    "What is the dipole moment of this molecule?",
    "Give the molecular dipole vector.",
    "Which way does the dipole point, and how strong is it?",
    "Predict the electric dipole moment.",
]
DIPOLE_TEMPLATES_HELDOUT = [
    "Report the total dipole of the structure.",
    "What dipole does this compound have?",
]
FORCE_TEMPLATES = [
    "What is the force on {d}?",
    "Give the force vector acting on {d}.",
    "Predict the force felt by {d}.",
    "How is {d} being pushed?",
]
FORCE_TEMPLATES_HELDOUT = [
    "Report the net force on {d}.",
    "Which direction is {d} pulled, and how hard?",
]

MAX_FORCE = 0.5  # hartree/bohr. Skip molecules where the random displacement made a huge force.


def describe_neighbors(atom):
    """'bonded to two carbons and one hydrogen'"""
    counts = Counter(n.GetAtomicNum() for n in atom.GetNeighbors())
    parts = []
    for z in sorted(counts):  # fixed order: H, C, N, O, F
        k = counts[z]
        word = NAME[z] if k == 1 else NAME_PLURAL[z]
        parts.append(f"{COUNT_WORD.get(k, str(k))} {word}")
    if not parts:
        return "with no bonds"
    if len(parts) == 1:
        return "bonded to " + parts[0]
    return "bonded to " + ", ".join(parts[:-1]) + " and " + parts[-1]


def unique_atom_descriptions(mol):
    """Return {atom_index: description} for atoms that can be described uniquely."""
    atoms = list(mol.GetAtoms())
    short = {a.GetIdx(): f"the {NAME[a.GetAtomicNum()]} atom {describe_neighbors(a)}" for a in atoms}
    counts = Counter(short.values())
    result = {}
    for a in atoms:
        i = a.GetIdx()
        if counts[short[i]] == 1:
            result[i] = short[i]
    # second try: add ring information for atoms that were not unique
    ring = {}
    for a in atoms:
        i = a.GetIdx()
        if i in result:
            continue
        where = "in a ring" if a.IsInRing() else "not in a ring"
        ring[i] = f"the {NAME[a.GetAtomicNum()]} atom {where} {describe_neighbors(a)}"
    ring_counts = Counter(ring.values())
    all_long = Counter(list(ring.values()) + list(result.values()))
    for i, d in ring.items():
        if ring_counts[d] == 1 and all_long[d] == 1:
            result[i] = d
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", default="data/gdb9.sdf")
    parser.add_argument("--labels", default="data/qm9_xtb.pkl")
    parser.add_argument("--out", default="data/tensorqa_v0.json")
    parser.add_argument("--force_questions_per_mol", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    labels = pickle.load(open(args.labels, "rb"))
    supplier = Chem.SDMolSupplier(args.sdf, removeHs=False, sanitize=True)

    # split molecules 80 / 10 / 10
    mol_ids = list(range(len(labels)))
    rng.shuffle(mol_ids)
    n = len(mol_ids)
    split_of = {}
    for k, m in enumerate(mol_ids):
        split_of[m] = "train" if k < 0.8 * n else ("val" if k < 0.9 * n else "test")

    questions = []
    skipped = 0
    for m, lab in enumerate(labels):
        if np.abs(lab["forces"]).max() > MAX_FORCE:
            skipped += 1
            continue
        mol = supplier[lab["qm9_index"]]
        assert [a.GetAtomicNum() for a in mol.GetAtoms()] == list(lab["numbers"])
        split = split_of[m]
        descriptions = unique_atom_descriptions(mol)
        force_atoms = rng.sample(sorted(descriptions), min(args.force_questions_per_mol, len(descriptions)))

        # normal templates for every split
        questions.append(dict(mol=m, split=split, task="dipole", atom=-1,
                              question=rng.choice(DIPOLE_TEMPLATES)))
        for i in force_atoms:
            questions.append(dict(mol=m, split=split, task="force", atom=i,
                                  question=rng.choice(FORCE_TEMPLATES).format(d=descriptions[i])))
        # held-out wordings, test molecules only
        if split == "test":
            questions.append(dict(mol=m, split="test_heldout", task="dipole", atom=-1,
                                  question=rng.choice(DIPOLE_TEMPLATES_HELDOUT)))
            for i in force_atoms:
                questions.append(dict(mol=m, split="test_heldout", task="force", atom=i,
                                      question=rng.choice(FORCE_TEMPLATES_HELDOUT).format(d=descriptions[i])))

    json.dump(questions, open(args.out, "w"), indent=0)
    print("skipped molecules (huge force):", skipped)
    print("questions:", len(questions))
    print(Counter((q["split"], q["task"]) for q in questions))
    print("unique question strings:", len({q["question"] for q in questions}))
    for q in questions[:6]:
        print(" ", q["task"], "|", q["question"])


if __name__ == "__main__":
    main()

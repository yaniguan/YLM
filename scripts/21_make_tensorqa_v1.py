"""TensorQA-v1: leak-aware splits, free-form wordings, systematic-generalisation test sets.

Tasks (all answers are 3-vectors, hartree/bohr, GFN2-xTB):
  force   depth 1  F_A
  diff    depth 2  F_A - F_B
  proj    depth 2  (F_A . u) u, u = unit(r_B - r_A)   (signed AXIS projection: points away from B if F_A . u < 0)

Splits (rows carry `split`):
  train, val           molecules of the v1 train/val split (scripts/20_make_splits.py), training wordings
  test                 held-out molecules, training wordings
  test_heldout         the same questions as `test`, re-worded in the held-out style families (new wording)
  scaf, scaf_heldout   molecules whose Bemis-Murcko scaffold never occurs in train/val/test (new scaffold)
  comb                 held-out molecules, pairs whose (task, element of atom A) never occurs in train/val:
                       (proj, O) and (diff, N)   (unseen entity x operator combination)
  cf                   counterfactual groups on test molecules (`kind`: swap = {D(A,B), D(B,A)};
                       chain = {D(A,B), D(B,C), D(A,C)}), same wording inside a group

Wording: LLM-generated style families (data/paraphrases.json, scripts/19). Training uses textbook/terse/technical/
indirect; chatty and inverted appear only in *_heldout. Every row records its style family.

  python scripts/21_make_tensorqa_v1.py        # -> data/tensorqa_v1.json
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

HELD_COMBOS = {("proj", 8), ("diff", 7)}   # (task, atomic number of A)


def target_of(task, forces, pos, a, b):
    if task == "force":
        return forces[a]
    if task == "diff":
        return forces[a] - forces[b]
    u = pos[b] - pos[a]
    u = u / np.linalg.norm(u)
    return (forces[a] @ u) * u


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sdf", default="data/gdb9.sdf")
    p.add_argument("--labels", default="data/qm9_xtb.pkl")
    p.add_argument("--splits", default="data/splits_v1.json")
    p.add_argument("--paraphrases", default="data/paraphrases.json")
    p.add_argument("--out", default="data/tensorqa_v1.json")
    p.add_argument("--per_task", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    labels = pickle.load(open(args.labels, "rb"))
    split_of = {int(k): v for k, v in json.load(open(args.splits)).items()}
    P = json.load(open(args.paraphrases))
    held_styles = set(P["heldout_styles"])
    styles = {t: {"train": {s: v for s, v in d.items() if s not in held_styles and v},
                  "held": {s: v for s, v in d.items() if s in held_styles and v}} for t, d in P["templates"].items()}
    supplier = Chem.SDMolSupplier(args.sdf, removeHs=False, sanitize=True)

    def pick(task, kind):
        style = rng.choice(sorted(styles[task][kind]))
        return style, rng.choice(styles[task][kind][style])

    out, gid = [], 0

    def add(m, split, task, atoms, desc, target, style, tmpl, **extra):
        a, b = atoms
        q = tmpl.format(a=desc[a], b=desc[b] if b >= 0 else "")
        out.append(dict(mol=m, split=split, task=task, atoms=[a, b], target=np.asarray(target).tolist(),
                        question=q, style=style, **extra))

    for m in sorted(split_of):
        lab = labels[m]
        if np.abs(lab["forces"]).max() > mkq.MAX_FORCE:
            continue
        sp = split_of[m]
        desc = mkq.unique_atom_descriptions(supplier[lab["qm9_index"]])
        atoms = sorted(desc)
        pos, forces, Z = np.asarray(lab["pos"], float), np.asarray(lab["forces"], float), lab["numbers"]
        train_like = {"train": ("train", None), "val": ("val", None),
                      "test": ("test", "test_heldout"), "scaf_test": ("scaf", "scaf_heldout")}[sp]

        def emit(task, ab, split_name, held_name):
            tgt = target_of(task, forces, pos, *ab) if ab[1] >= 0 or task == "force" else None
            if np.abs(tgt).max() > 2 * mkq.MAX_FORCE:
                return
            style, t = pick(task, "train")
            add(m, split_name, task, ab, desc, tgt, style, t)
            if held_name:
                style, t = pick(task, "held")
                add(m, held_name, task, ab, desc, tgt, style, t)

        for a in rng.sample(atoms, min(args.per_task, len(atoms))):
            emit("force", (a, -1), *train_like)
        if len(atoms) < 2:
            continue
        for task in ["diff", "proj"]:
            n = 0
            for _ in range(20 * args.per_task):
                if n >= args.per_task:
                    break
                a, b = rng.sample(atoms, 2)
                held = (task, int(Z[a])) in HELD_COMBOS
                if sp in ("train", "val") and held:
                    continue
                if sp == "test" and held:
                    emit(task, (a, b), "comb", None)
                else:
                    emit(task, (a, b), *train_like)
                n += 1
            if sp == "test":   # make sure the held-out combinations get examples
                for z in {zz for t, zz in HELD_COMBOS if t == task}:
                    cands = [x for x in atoms if Z[x] == z]
                    if cands and len(atoms) > 1:
                        a = rng.choice(cands)
                        b = rng.choice([x for x in atoms if x != a])
                        emit(task, (a, b), "comb", None)
        if sp == "test" and len(atoms) >= 3:   # counterfactual groups, same wording inside a group
            a, b, c = rng.sample(atoms, 3)
            style, t = pick("diff", "train")
            for kind, pairs in [("chain", [(a, b), (b, c), (a, c)]), ("swap", [(a, b), (b, a)])]:
                gid += 1
                for k, ab in enumerate(pairs):
                    add(m, "cf", "diff", ab, desc, target_of("diff", forces, pos, *ab), style, t, kind=kind, gid=gid, k=k)

    json.dump(out, open(args.out, "w"))
    print("rows:", len(out), "| unique questions:", len({q["question"] for q in out}))
    c = Counter((q["split"], q["task"]) for q in out)
    for k in sorted(c):
        print(" ", k, c[k])
    print("held-out style rows:", Counter(q["style"] for q in out if q["split"].endswith("heldout")))


if __name__ == "__main__":
    main()

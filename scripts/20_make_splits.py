"""Leak-aware molecule splits for TensorQA-v1.

The v0 split is a random 80/10/10 over molecule indices. This script measures how much that leaks and builds a
stricter one:
  * molecules with the same canonical graph (non-isomeric SMILES of the heavy-atom graph) are one group and never
    cross a split boundary;
  * a scaffold split: whole Bemis-Murcko scaffold groups are held out as `scaf_test` (never in train/val/test);
  * the remaining groups are split 80/10/10 into train/val/test.

  python scripts/20_make_splits.py      # -> data/splits_v1.json {mol: split}, prints the v0 leakage numbers
"""
import argparse
import json
import pickle
import random
from collections import Counter, defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sdf", default="data/gdb9.sdf")
    p.add_argument("--labels", default="data/qm9_xtb.pkl")
    p.add_argument("--v0", default="data/tensorqa_v0.json")
    p.add_argument("--out", default="data/splits_v1.json")
    p.add_argument("--scaf_frac", type=float, default=0.10, help="fraction of molecules in held-out scaffold groups")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    labels = pickle.load(open(args.labels, "rb"))
    supplier = Chem.SDMolSupplier(args.sdf, removeHs=False, sanitize=True)
    graph, scaf = {}, {}
    for m, lab in enumerate(labels):
        mol = Chem.RemoveHs(supplier[lab["qm9_index"]])
        graph[m] = Chem.MolToSmiles(mol, isomericSmiles=False)
        scaf[m] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

    # --- how leaky is the v0 split? ---
    v0 = {q["mol"]: q["split"] for q in json.load(open(args.v0)) if q["split"] != "test_heldout"}
    by_graph = defaultdict(set)
    for m, sp in v0.items():
        by_graph[graph[m]].add(sp)
    shared = sum(1 for s in by_graph.values() if len(s) > 1)
    test_m = [m for m, sp in v0.items() if sp == "test"]
    train_scaf = {scaf[m] for m, sp in v0.items() if sp == "train"}
    seen = sum(scaf[m] in train_scaf for m in test_m)
    acyclic = sum(scaf[m] == "" for m in test_m)
    print(f"v0: {shared} canonical graphs appear in more than one split")
    print(f"v0: {seen}/{len(test_m)} = {seen / len(test_m):.1%} of test molecules have a scaffold seen in train "
          f"({acyclic} test molecules are acyclic, scaffold '')")

    # --- v1 split ---
    rng = random.Random(args.seed)
    groups = defaultdict(list)           # graph -> molecules
    for m in graph:
        groups[graph[m]].append(m)
    scaf_groups = defaultdict(list)      # scaffold -> graphs (cyclic scaffolds only: acyclic is one huge generic group)
    for g, ms in groups.items():
        scaf_groups[scaf[ms[0]]].append(g)
    cyc = [s for s in scaf_groups if s != ""]
    rng.shuffle(cyc)
    held, n_held, target = set(), 0, args.scaf_frac * len(graph)
    for s in cyc:
        if scaf_groups[s] and n_held >= target:
            break
        # small scaffold groups first would bias toward rare scaffolds, so we take them in random order
        held.add(s)
        n_held += sum(len(groups[g]) for g in scaf_groups[s])
    split = {}
    rest = []
    for g, ms in groups.items():
        if scaf[ms[0]] in held:
            for m in ms:
                split[m] = "scaf_test"
        else:
            rest.append(g)
    rng.shuffle(rest)
    n_rest = sum(len(groups[g]) for g in rest)
    c = 0
    for g in rest:
        sp = "train" if c < 0.8 * n_rest else "val" if c < 0.9 * n_rest else "test"
        for m in groups[g]:
            split[m] = sp
        c += len(groups[g])
    json.dump({str(m): sp for m, sp in split.items()}, open(args.out, "w"))
    print("v1 split sizes:", dict(Counter(split.values())))
    tr = {scaf[m] for m, sp in split.items() if sp == "train"}
    for sp in ["val", "test", "scaf_test"]:
        ms = [m for m, s in split.items() if s == sp]
        print(f"v1 {sp}: scaffold seen in train {sum(scaf[m] in tr for m in ms) / len(ms):.1%}; "
              f"graphs shared with train {sum(graph[m] in {graph[x] for x in split if split[x] == 'train'} for m in ms)}")


if __name__ == "__main__":
    main()

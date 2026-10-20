"""Run the rule parser on every compositional question and cache the result (needs RDKit, so run locally).

Output: data/parsed_comp.json = [{"task": "diff"|"proj"|null, "atoms": [a, b]}, ...] aligned to tensorqa_comp.json.
Also prints parser accuracy against the generator's ground truth, per split and task.

  python scripts/16_parse_comp.py
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict

from rdkit import Chem, RDLogger

sys.path.insert(0, ".")
from src.parse_question import resolve_program  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sdf", default="data/gdb9.sdf")
    p.add_argument("--labels", default="data/qm9_xtb.pkl")
    p.add_argument("--questions", default="data/tensorqa_comp.json")
    p.add_argument("--out", default="data/parsed_comp.json")
    args = p.parse_args()
    labels = pickle.load(open(args.labels, "rb"))
    qs = json.load(open(args.questions))
    supplier = Chem.SDMolSupplier(args.sdf, removeHs=False, sanitize=True)
    mols, out, ok, by_style = {}, [], defaultdict(list), defaultdict(list)
    for q in qs:
        m = q["mol"]
        if m not in mols:
            mols[m] = supplier[labels[m]["qm9_index"]]
        task, a, b = resolve_program(q["question"], mols[m])
        out.append(dict(task=task, atoms=[a, b]))
        good = task == q["task"] and [a, b] == q["atoms"]
        ok[(q["split"], q["task"])].append(good)
        if "style" in q:
            by_style[q["style"]].append(good)
    json.dump(out, open(args.out, "w"))
    for k in sorted(ok):
        print(k, f"parser accuracy {sum(ok[k]) / len(ok[k]):.4f} (n={len(ok[k])})")
    for k in sorted(by_style):
        print("style", k, f"{sum(by_style[k]) / len(by_style[k]):.4f} (n={len(by_style[k])})")


if __name__ == "__main__":
    main()
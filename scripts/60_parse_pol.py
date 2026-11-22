"""Run the rule parser on polarizability questions and report accuracy (needs RDKit). -> data/parsed_pol.json"""
import argparse
import importlib.util
import json
import os
import pickle
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from src.parse_question import resolve_pol  # noqa: E402

spec = importlib.util.spec_from_file_location("mk", os.path.join(HERE, "58_make_pol_data.py"))
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/qm7b_pol.pkl")
    ap.add_argument("--questions", default="data/tensorqa_pol.json")
    ap.add_argument("--out", default="data/parsed_pol.json")
    args = ap.parse_args()
    labels = pickle.load(open(args.labels, "rb"))
    qs = json.load(open(args.questions))
    mols, out, ok, by_style = {}, [], defaultdict(list), defaultdict(list)
    for q in qs:
        m = q["mol"]
        if m not in mols:
            mols[m] = mk.connectivity_mol(labels[m]["numbers"], labels[m]["pos"])
        task, a, b = resolve_pol(q["question"], mols[m])
        out.append(dict(task=task, atoms=[a, b]))
        good = task == q["task"] and [a, b] == q["atoms"]
        ok[(q["split"], q["task"])].append(good)
        by_style[q["style"]].append(good)
    json.dump(out, open(args.out, "w"))
    for k in sorted(ok):
        print(k, f"{sum(ok[k]) / len(ok[k]):.4f} (n={len(ok[k])})")
    for k in sorted(by_style):
        print("style", k, f"{sum(by_style[k]) / len(by_style[k]):.4f}")


if __name__ == "__main__":
    main()

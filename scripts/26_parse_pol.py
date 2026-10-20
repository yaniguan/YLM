"""Rule-parser the polarizability questions and report accuracy (needs RDKit).  -> data/parsed_pol.json"""
import importlib.util
import json
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from src.parse_question import resolve_pol  # noqa: E402

spec = importlib.util.spec_from_file_location("mk", "scripts/24_make_pol_data.py")
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)

labels = pickle.load(open("data/qm7b_pol.pkl", "rb"))
qs = json.load(open("data/tensorqa_pol.json"))
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
json.dump(out, open("data/parsed_pol.json", "w"))
for k in sorted(ok):
    print(k, f"{sum(ok[k]) / len(ok[k]):.4f} (n={len(ok[k])})")
for k in sorted(by_style):
    print("style", k, f"{sum(by_style[k]) / len(by_style[k]):.4f}")

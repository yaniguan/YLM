"""Assemble the own benchmark (ChemProto-Twins) from reviewed scenario files into item instances."""
import json, glob, hashlib, sys
from rdkit import Chem
from extract_chems import resolve
from paths import DATA

def valid(smi):
    return smi and Chem.MolFromSmiles(smi) is not None

def main(scen_files, out=None):
    out = out or (DATA / "items_own.jsonl")
    sc = []
    for f in scen_files: sc += json.load(open(f))
    seen = set(); rows = []; bad_smiles = 0
    for s in sc:
        sid = s["scenario_id"]
        if sid in seen: continue
        seen.add(sid)
        mols = []
        for m in s.get("molecules", []):
            smi = m.get("smiles")
            if not valid(smi):
                smi = resolve(m["name"]); bad_smiles += 1
            if valid(smi): mols.append({"name": m["name"], "smiles": smi})
        split = "dev" if int(hashlib.md5(sid.encode()).hexdigest(), 16) % 5 == 0 else "test"
        base = {"bench": "own", "label": s["label"], "molecules": mols, "context": s.get("context") or {},
                "split": split, "group": sid, "lablean_expressible": bool(s.get("lablean_expressible")),
                "dependent": s["label"] != "none"}
        for suf, key, is_err in [("c", "protocol_correct", False), ("b", "protocol_broken", True), ("d", "protocol_decoy", False)]:
            r = dict(base, id=f"{sid}-{suf}", task="error", protocol=s[key], is_error=is_err, variant=suf,
                     broken_step=s.get("broken_step") if is_err else None)
            rows.append(r)
        q = s["mcq"]
        rows.append(dict(base, id=f"{sid}-q", task=q["task"], stem=q["question"], choices=q["choices"],
                         answer_index=int(q["answer_index"]), variant="q"))
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(len(seen), "scenarios", len(rows), "items", "re-resolved smiles:", bad_smiles)

if __name__ == "__main__":
    main(sys.argv[1:])

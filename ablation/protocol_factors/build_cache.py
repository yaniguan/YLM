"""Precompute xTB (Q, S) and MoLFormer predictions (F) for all molecules in the given item files."""
import json, sys, pickle, numpy as np
from multiprocessing import Pool
from pathlib import Path
import features as FE
from render import ctx_of
from paths import CACHE

def _xtb(args):
    smi, solv = args
    try: FE.qm_and_solv(smi, solv)
    except Exception as e: print("xtb fail", smi, e, flush=True)
    return smi

def main(item_files):
    items = []
    for f in item_files: items += [json.loads(l) for l in open(f)]
    jobs = set(); smis = set()
    for it in items:
        c = ctx_of(it)
        for m in it.get("molecules", []):
            if m.get("smiles") and FE.Chem.MolFromSmiles(m["smiles"]):
                jobs.add((m["smiles"], c.get("solvent_key"))); smis.add(m["smiles"])
    print(len(items), "items", len(smis), "molecules", len(jobs), "xtb jobs", flush=True)
    fmf = CACHE / "fm_pred.json"; fm = json.loads(fmf.read_text()) if fmf.exists() else {}
    todo = sorted(s for s in smis if s not in fm)
    if todo:
        from fm import predict
        X, p = predict(todo)
        emb = CACHE / "fm_emb.pkl"; E = pickle.load(open(emb, "rb")) if emb.exists() else {}
        for i, s in enumerate(todo):
            fm[s] = {k: float(v[i]) for k, v in p.items()}; E[s] = X[i].astype(np.float32)
        pickle.dump(E, open(emb, "wb")); fmf.write_text(json.dumps(fm))
    with Pool(8) as pool:
        for i, _ in enumerate(pool.imap_unordered(_xtb, sorted(jobs, key=str))):
            if i % 50 == 0: print("xtb", i, flush=True)

if __name__ == "__main__":
    main(sys.argv[1:])

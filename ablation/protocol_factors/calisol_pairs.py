"""Build CALiSol-23 pairwise items: which of two formulations has higher ionic conductivity.
Same publication (doi), |T1-T2| <= 1 K, >= 10% relative difference. Split by salt-solvent system."""
import json, random, numpy as np, pandas as pd, hashlib
from pathlib import Path
import os
from paths import DATA
SRC = Path(os.environ.get("CALISOL_DIR", "data/calisol"))  # calisol23.csv + calisol_smiles.csv (CC-BY 4.0)

def load():
    df = pd.read_csv(SRC/"calisol23.csv")
    solv_cols = list(df.columns[7:])
    smi = {}
    for line in open(SRC/"calisol_smiles.csv", encoding="utf-8-sig"):
        parts = [p.strip() for p in line.split(";")]
        if len(parts) >= 3:
            if parts[0].startswith("Li"): smi[parts[0]] = parts[2]
            else:
                ab = parts[1] if parts[2] else parts[0]; s = parts[2] or parts[1]
                smi[ab] = s; smi[parts[0]] = s
    for ab, full in [("EC","Ethylene carbonate"),("PC","Propylene carbonate"),("DMC","Dimethyl carbonate"),("EMC","Ethyl Methyl Carbonate")]:
        smi[ab] = smi[full]
    smi["LiTFSI"] = smi["LiN(CF3SO2)2"]  # use the ionic form
    smi["LiPDI"] = "[Li+].C(#N)C1=C(N=C([N-]1)C(C(F)(F)F)(F)F)C#N"  # add Li+ (source lists the neutral acid)
    df["salt"] = df.salt.str.strip()
    df = df[(df.k > 0) & df["T"].between(200, 400) & df.c.notna()].copy()
    return df, solv_cols, smi

def system_key(r, solv_cols):
    return r.salt + "|" + "+".join(c for c in solv_cols if r[c] > 0)

def describe(r, solv_cols):
    comps = [(c, r[c]) for c in solv_cols if r[c] > 0]
    tot = sum(v for _, v in comps)
    rt = {"w": "weight", "v": "volume", "m": "mole"}.get(str(r["solvent ratio type"]), str(r["solvent ratio type"]))
    sv = ":".join(c for c, _ in comps) + " " + ":".join(f"{v/tot:.2f}" for _, v in comps) + f" ({rt} ratio)"
    return f"{r.c:g} {r['c units']} {r.salt} in {sv}, at {r['T']-273.15:.0f} C", comps

def build(seed=0, n_test=300, n_train=3000, n_dev=300):
    df, solv_cols, smi = load()
    df["sys"] = [system_key(r, solv_cols) for _, r in df.iterrows()]
    systems = sorted(df.sys.unique()); rng = random.Random(seed); rng.shuffle(systems)
    k = len(systems); test_sys = set(systems[: k // 5]); dev_sys = set(systems[k // 5: k // 5 + k // 10])
    df["split"] = ["test" if s in test_sys else "dev" if s in dev_sys else "train" for s in df.sys]
    items = {"train": [], "dev": [], "test": []}
    for split in items:
        d = df[df.split == split]
        for doi, g in d.groupby("doi"):
            g = g.reset_index(drop=True); T = g["T"].values; K = g.k.values
            desc = [describe(g.loc[i], solv_cols)[0] for i in range(len(g))]
            det_seed = int(hashlib.md5(str(doi).encode("utf-8")).hexdigest()[:8], 16) % (10**6)
            rr = random.Random(det_seed); cand = []
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    if abs(T[i] - T[j]) <= 1.0 and abs(K[i] - K[j]) / max(K[i], K[j]) >= 0.10 and desc[i] != desc[j]:
                        cand.append((i, j))
            rr.shuffle(cand)
            for i, j in cand[:400]: items[split].append((g.loc[i], g.loc[j]))
    out = {}
    for split, n in [("test", n_test), ("train", n_train), ("dev", n_dev)]:
        pairs = items[split]; rng2 = random.Random(seed + 1); rng2.shuffle(pairs)
        # cap pairs per doi so one paper does not dominate
        per, sel = {}, []
        cap = max(3, n // 6)
        for a, b in pairs:
            if per.get(a.doi, 0) >= cap: continue
            per[a.doi] = per.get(a.doi, 0) + 1; sel.append((a, b))
            if len(sel) >= n: break
        rows = []
        for t, (a, b) in enumerate(sel):
            if rng2.random() < 0.5: a, b = b, a
            da, ca = describe(a, solv_cols); db, cb = describe(b, solv_cols)
            def form(r, comps):
                return {"salt": r.salt, "salt_smiles": smi.get(r.salt), "c": float(r.c), "c_units": r["c units"],
                        "T_K": float(r["T"]), "solvents": [{"name": c, "frac": float(v), "smiles": smi.get(c)} for c, v in comps],
                        "ratio_type": str(r["solvent ratio type"])}
            rows.append({"id": f"calisol-{split}-{t:04d}", "bench": "calisol", "task": "reasoning", "label": "solvation",
                         "question": f"Which liquid electrolyte has the higher ionic conductivity?\nA) {da}\nB) {db}",
                         "choices": ["A", "B"], "answer_index": 0 if a.k > b.k else 1,
                         "formulations": [form(a, ca), form(b, cb)], "doi": a.doi,
                         "k_mS_cm": [float(a.k), float(b.k)], "split": split})
        out[split] = rows
    return out

if __name__ == "__main__":
    out = build()
    for s, rows in out.items():
        with open(DATA / f"calisol_{s}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        print(s, len(rows), "frac A:", np.mean([r["answer_index"] == 0 for r in rows]).round(3),
              "dois:", len({r["doi"] for r in rows}))

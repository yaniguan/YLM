"""CALiSol pair probes: info layers (Q, F, S, M; 16 subsets) x modality (text, native, both) x probe (logreg, kNN).
Text = MiniLM sentence embedding of the formulation written out in words (with the same layer facts as text).
Native = the same facts as a standardized number vector. Both = concatenation.
Five seeds; each seed trains on a random half of the train pairs. Test pairs come from held-out salt-solvent systems."""
import json, itertools, sys, numpy as np, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import n2_calisol as N
from paths import DATA, RESULTS

torch.set_num_threads(4)
LAYERS = ["Q", "F", "S", "M"]
KEYS = {"base": ["T_C", "conc", "molal"],
        "Q": ["anion_homo", "anion_lumo", "anion_gap", "anion_vip", "solv_dipole", "solv_gap", "solv_homo"],
        "F": ["solv_logS", "solv_logD", "solv_logP", "solv_tpsa"],
        "S": ["anion_dGw", "solv_dielectric", "solv_viscosity"],
        "M": ["solv_mp", "solv_bp", "T_minus_mp"]}
NAMES = {"T_C": "temperature C", "conc": "salt concentration", "molal": "molal units flag",
         "T_minus_mp": "temperature above solvent mp (C)"}
SEEDS = [0, 1, 2, 3, 4]

def enrich(x):
    x = dict(x)
    if x.get("solv_mp") is not None: x["T_minus_mp"] = x["T_C"] - x["solv_mp"]
    return x

def keys_for(sub):
    return KEYS["base"] + [k for L in LAYERS if L in sub for k in KEYS[L]]

def text_for(desc, x, sub):
    parts = [desc]
    for k in keys_for(sub)[3:]:
        if x.get(k) is not None:
            parts.append(f"{NAMES.get(k, k.replace('_', ' '))} {x[k]:.3g}")
    return "; ".join(parts)

class Enc:
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tok = AutoTokenizer.from_pretrained(name); self.m = AutoModel.from_pretrained(name).eval(); self.cache = {}
    @torch.no_grad()
    def __call__(self, texts):
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(new), 128):
            b = self.tok(new[i:i + 128], padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = self.m(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
            e = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).numpy()
            for t, v in zip(new[i:i + 128], e): self.cache[t] = v
        return np.array([self.cache[t] for t in texts])

VOCAB = []

def ident(f):
    """Identity one-hot: salt and solvent weight fractions (the native twin of the words in the text)."""
    tot = sum(s["frac"] for s in f["solvents"]); d = {"salt:" + f["salt"]: 1.0}
    for s in f["solvents"]: d["solv:" + s["name"]] = d.get("solv:" + s["name"], 0) + s["frac"] / tot
    return np.array([d.get(k, 0.0) for k in VOCAB])

def native(rows, sub, mu_sd):
    ks = keys_for(sub); mu, sd = mu_sd
    def v(x, f): 
        a = np.array([x.get(k) if x.get(k) is not None else np.nan for k in ks], float)
        miss = np.isnan(a).astype(float); a = np.nan_to_num((a - mu[ks].values) / sd[ks].values)
        return np.concatenate([a, miss, ident(f)])
    A = np.array([v(r["xa"], r["fa"]) for r in rows]); B = np.array([v(r["xb"], r["fb"]) for r in rows])
    return np.hstack([A - B, A + B])

def textual(rows, sub, enc):
    A = enc([text_for(r["da"], r["xa"], sub) for r in rows]); B = enc([text_for(r["db"], r["xb"], sub) for r in rows])
    return np.hstack([A - B, A + B])

def flip(X, y):
    """Augment with A/B swapped (diff part flips sign) so probes cannot use answer position."""
    h = X.shape[1] // 2
    return np.vstack([X, np.hstack([-X[:, :h], X[:, h:]])]), np.concatenate([y, 1 - y])

def main(out=None):
    import pandas as pd
    out = out or str(RESULTS / "probe_calisol.json")
    tr = N.load("train"); te = N.load("test")
    for split, rows in (("train", tr), ("test", te)):
        raw = [json.loads(l) for l in open(DATA / f"calisol_{split}.jsonl")]
        for r, q in zip(rows, raw):
            assert r["id"] == q["id"]; r["fa"], r["fb"] = q["formulations"]
    VOCAB.extend(sorted({k for r in tr for f in (r["fa"], r["fb"]) for k in
                         ["salt:" + f["salt"]] + ["solv:" + s["name"] for s in f["solvents"]]}))
    for r in tr + te: r["xa"] = enrich(r["xa"]); r["xb"] = enrich(r["xb"])
    allk = sorted({k for v in KEYS.values() for k in v})
    df = pd.DataFrame([{k: r[s].get(k) for k in allk} for r in tr for s in ("xa", "xb")], dtype=float)
    mu_sd = (df.mean(), df.std() + 1e-6)
    enc = Enc(); ytr = np.array([r["y"] for r in tr]); yte = np.array([r["y"] for r in te])
    test_ids = [r["id"] for r in te]
    res = {"test_ids": test_ids, "y": yte.tolist(), "cells": [], "seeds": SEEDS}
    have = set()
    if Path(out).exists():
        old = json.loads(Path(out).read_text())
        if old.get("test_ids") == test_ids:
            res = old
            res.setdefault("seeds", SEEDS)
            have = {(c["layers"], c["modality"], c["probe"], c["seed"]) for c in res["cells"]}
            print(f"resume: {len(have)} cell-seeds already stored", flush=True)
    for r_ in range(5):
        for sub in itertools.combinations(LAYERS, r_):
            sub = set(sub)
            layer_name = "".join(l for l in LAYERS if l in sub) or "none"
            need = [(mod, probe, seed) for mod in ("native", "text", "both") for probe in ("logreg", "knn") for seed in SEEDS
                    if (layer_name, mod, probe, seed) not in have]
            if not need:
                print(sorted(sub) or "none", "already done", flush=True)
                continue
            feats = {}
            feats["native"] = (native(tr, sub, mu_sd), native(te, sub, mu_sd))
            feats["text"] = (textual(tr, sub, enc), textual(te, sub, enc))
            feats["both"] = (np.hstack([feats["native"][0], feats["text"][0]]), np.hstack([feats["native"][1], feats["text"][1]]))
            for mod, (Xtr, Xte) in feats.items():
                for probe in ["logreg", "knn"]:
                    for seed in SEEDS:
                        if (layer_name, mod, probe, seed) in have:
                            continue
                        rng = np.random.default_rng(seed); idx = rng.choice(len(tr), len(tr) // 2, replace=False)
                        X, y = flip(Xtr[idx], ytr[idx]); sc = StandardScaler().fit(X)
                        clf = (LogisticRegression(max_iter=3000, C=0.5) if probe == "logreg"
                               else KNeighborsClassifier(n_neighbors=15))
                        clf.fit(sc.transform(X), y)
                        pred = clf.predict(sc.transform(Xte))
                        res["cells"].append({"layers": "".join(l for l in LAYERS if l in sub) or "none", "modality": mod,
                                             "probe": probe, "seed": seed, "correct": (pred == yte).astype(int).tolist()})
                    accs = [np.mean(c["correct"]) for c in res["cells"][-len(SEEDS):]]
                    print(sorted(sub), mod, probe, f"{np.mean(accs):.3f} +- {np.std(accs):.3f}", flush=True)
            Path(out).write_text(json.dumps(res))

if __name__ == "__main__":
    main(*sys.argv[1:])

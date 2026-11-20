"""Molecular foundation-model features: MoLFormer-XL (Apache-2.0) mean-pooled embeddings
+ ridge heads trained on public MoleculeNet ESOL (logS) and Lipophilicity (logD7.4)."""
import json, pickle, numpy as np, pandas as pd, torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
NAME = "ibm-research/MoLFormer-XL-both-10pct"
HERE = Path(__file__).parent
_tok = _m = None

def embed(smiles, bs=64):
    global _tok, _m
    if _m is None:
        _tok = AutoTokenizer.from_pretrained(NAME, trust_remote_code=True)
        _m = AutoModel.from_pretrained(NAME, deterministic_eval=True, trust_remote_code=True).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), bs):
            b = _tok(smiles[i:i+bs], padding=True, return_tensors="pt")
            h = _m(**b).last_hidden_state
            msk = b["attention_mask"].unsqueeze(-1)
            out.append(((h*msk).sum(1)/msk.sum(1)).numpy())
    return np.concatenate(out)

def train_heads(data_dir):
    res, heads = {}, {}
    for key, f, col in [("logS", "delaney-processed.csv", "measured log solubility in mols per litre"),
                        ("logD", "Lipophilicity.csv", "exp")]:
        df = pd.read_csv(Path(data_dir)/f); X = embed(df["smiles"].tolist()); y = df[col].values
        r = RidgeCV(alphas=np.logspace(-2, 3, 12))
        p = cross_val_predict(r, X, y, cv=KFold(5, shuffle=True, random_state=0))
        res[key] = {"n": len(y), "cv5_rmse": float(np.sqrt(np.mean((p-y)**2))),
                    "cv5_r2": float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2))}
        heads[key] = r.fit(X, y)
    pickle.dump(heads, open(HERE/"fm_heads.pkl", "wb"))
    json.dump(res, open(HERE/"fm_heads_cv.json", "w"), indent=1)
    return res

def predict(smiles):
    heads = pickle.load(open(HERE/"fm_heads.pkl", "rb"))
    X = embed(smiles)
    return X, {k: h.predict(X) for k, h in heads.items()}

if __name__ == "__main__":
    import sys; print(train_heads(sys.argv[1]))

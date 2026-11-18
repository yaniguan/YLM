"""Native-input test on CALiSol pairs with a frozen Qwen2.5-0.5B-Instruct (CPU).
Arms (same data, steps, optimizer; only the input path changes):
  N0  formulation text + 4 learned constant soft tokens per formulation (format/label-prior control)
  N1  N0 + the interpretable scalars rendered as text
  N2  formulation text + 4 soft tokens per formulation from an MLP projector on the scalars (+ PCA-32 MoLFormer embedding if --emb)
  N3  N1 text + N2 soft tokens
  N2-shuf  N2 with each formulation's features taken from a random other training formulation
Ceiling: logistic regression on the same feature differences."""
import json, sys, math, random, pickle, time, argparse, numpy as np, torch, torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import features as FE
from tables import ELECTROLYTE_SOLVENTS, SOLVENTS
from paths import DATA, CACHE, RESULTS
torch.set_num_threads(4); torch.manual_seed(0)
NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SCAL = ["T_C", "conc", "molal", "anion_homo", "anion_lumo", "anion_gap", "anion_vip", "anion_dGw",
        "solv_dielectric", "solv_viscosity", "solv_mp", "solv_bp", "solv_dipole", "solv_gap", "solv_homo",
        "solv_logS", "solv_logD", "solv_logP", "solv_tpsa", "frac_known_dielectric", "frac_known_viscosity"]

def xtb(smi):
    r = FE.qm_and_solv(smi, None); return r
def solvent_props(name, smi):
    e = eta = mp = bp = None
    k = FE.SOLV_BY_SMI.get(FE.canon(smi)) if smi else None
    if k: e, bp, mp, _ = SOLVENTS[k]
    v = ELECTROLYTE_SOLVENTS.get(name)
    if v:
        eta = v[1]
        if e is None: e, mp, bp = v[0], v[2], v[3]
    return e, eta, mp, bp

FM = EMB = None

def _fm():
    global FM, EMB
    if FM is None:
        FM = json.load(open(CACHE / "fm_pred.json"))
        EMB = pickle.load(open(CACHE / "fm_emb.pkl", "rb"))
    return FM, EMB

def featurize(f):
    """f: formulation dict from data/calisol_*.jsonl"""
    x = {"T_C": f["T_K"] - 273.15, "conc": f["c"], "molal": 1.0 if "kg" in f["c_units"] else 0.0}
    an = [k for k in f["salt_smiles"].split(".") if "Li" not in k]
    d = xtb(f["salt_smiles"]).get(an[0]) if an else None
    if d:
        x.update(anion_homo=d["homo_eV"], anion_lumo=d["lumo_eV"], anion_gap=d["gap_eV"], anion_vip=d["vIP_eV"],
                 anion_dGw=d.get("dGsolv_water_kcal"))
    tot = sum(s["frac"] for s in f["solvents"]); acc = {}; wk = {"e": 0, "eta": 0}
    for s in f["solvents"]:
        w = s["frac"] / tot; e, eta, mp, bp = solvent_props(s["name"], s["smiles"])
        dd = (xtb(s["smiles"]) or {}).get(s["smiles"]) or next(iter((xtb(s["smiles"]) or {None: None}).values()), None)
        rd = FE.rdkit_desc(s["smiles"]); fm = _fm()[0].get(s["smiles"], {})
        vals = {"solv_dielectric": e, "solv_viscosity": eta, "solv_mp": mp, "solv_bp": bp,
                "solv_dipole": dd and dd["dipole_D"], "solv_gap": dd and dd["gap_eV"], "solv_homo": dd and dd["homo_eV"],
                "solv_logS": fm.get("logS"), "solv_logD": fm.get("logD"), "solv_logP": rd["logP"], "solv_tpsa": rd["TPSA"]}
        for k, v in vals.items():
            if v is not None: acc.setdefault(k, [0.0, 0.0]); acc[k][0] += w * v; acc[k][1] += w
        if e is not None: wk["e"] += w
        if eta is not None: wk["eta"] += w
    for k, (s_, w_) in acc.items(): x[k] = s_ / w_ if w_ > 0 else None
    x["frac_known_dielectric"] = wk["e"]; x["frac_known_viscosity"] = wk["eta"]
    E = _fm()[1]; emb = sum((s["frac"] / tot) * E[s["smiles"]] for s in f["solvents"]) + E[f["salt_smiles"]]
    return x, emb

def text_of(x):
    nm = {"anion_homo": "anion HOMO (eV)", "anion_lumo": "anion LUMO (eV)", "anion_gap": "anion gap (eV)",
          "anion_vip": "anion vertical IP (eV)", "anion_dGw": "anion water solvation (kcal/mol)",
          "solv_dielectric": "solvent dielectric (mix avg)", "solv_viscosity": "solvent viscosity cP (mix avg)",
          "solv_mp": "solvent mp C (mix avg)", "solv_bp": "solvent bp C (mix avg)", "solv_dipole": "solvent dipole D (mix avg)",
          "solv_gap": "solvent gap eV (mix avg)", "solv_homo": "solvent HOMO eV (mix avg)", "solv_logS": "solvent logS (mix avg)",
          "solv_logD": "solvent logD (mix avg)", "solv_logP": "solvent logP (mix avg)", "solv_tpsa": "solvent TPSA (mix avg)"}
    return "; ".join(f"{nm[k]} {x[k]:.3g}" for k in nm if x.get(k) is not None)

def load(split):
    rows = [json.loads(l) for l in open(DATA / f"calisol_{split}.jsonl")]
    out = []
    for r in rows:
        fa, fb = r["formulations"]; xa, ea = featurize(fa); xb, eb = featurize(fb)
        q = r["question"].split("\n")
        out.append({"id": r["id"], "y": r["answer_index"], "da": q[1][3:], "db": q[2][3:], "xa": xa, "xb": xb, "ea": ea, "eb": eb})
    return out

class Std:
    def fit(self, rows):
        M = np.array([[r[s].get(k) if r[s].get(k) is not None else np.nan for k in SCAL] for r in rows for s in ("xa", "xb")], float)
        self.mu = np.nanmean(M, 0); self.sd = np.nanstd(M, 0) + 1e-6; return self
    def __call__(self, x):
        v = np.array([x.get(k) if x.get(k) is not None else np.nan for k in SCAL], float)
        miss = np.isnan(v).astype(float); v = np.nan_to_num((v - self.mu) / self.sd)
        return np.concatenate([v, miss])

def logistic_ceiling(tr, te, std, pca=None):
    from sklearn.linear_model import LogisticRegression
    def X(rows, emb):
        a = np.array([std(r["xa"]) - std(r["xb"]) for r in rows])
        if emb: a = np.hstack([a, pca.transform(np.array([r["ea"] for r in rows])) - pca.transform(np.array([r["eb"] for r in rows]))])
        return a
    res = {}
    for emb in [False, True]:
        if emb and pca is None: continue
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X(tr, emb), [r["y"] for r in tr])
        res["logreg" + ("+emb" if emb else "")] = [int(p == r["y"]) for p, r in zip(clf.predict(X(te, emb)), te)]
    return res

class Proj(nn.Module):
    def __init__(self, din, d, k=4):
        super().__init__(); self.k, self.d = k, d
        self.net = nn.Sequential(nn.Linear(din, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, k * d))
    def forward(self, v): return self.net(v).view(-1, self.k, self.d)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", default="N0,N1,N2,N3,N2-shuf")
    ap.add_argument("--ntrain", type=int, default=1200); ap.add_argument("--emb", action="store_true")
    ap.add_argument("--out", default=""); a = ap.parse_args(); a.out = a.out or str(RESULTS / "n2_calisol.json")
    tr = load("train"); te = load("test"); random.Random(0).shuffle(tr); tr = tr[: a.ntrain]
    std = Std().fit(tr)
    from sklearn.decomposition import PCA
    pca = PCA(32, random_state=0).fit(np.array([r[k] for r in tr for k in ("ea", "eb")]))
    res = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    res.update({k: v for k, v in logistic_ceiling(tr, te, std, pca).items()})
    res["test_ids"] = [r["id"] for r in te]
    tok = AutoTokenizer.from_pretrained(NAME); lm = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
    for p in lm.parameters(): p.requires_grad_(False)
    E = lm.get_input_embeddings(); d = E.weight.shape[1]; rms = E.weight.pow(2).mean().sqrt().item()
    idA, idB = tok("A", add_special_tokens=False).input_ids[0], tok("B", add_special_tokens=False).input_ids[0]
    def vec(r, side, shuf_pool=None):
        x, e = (r["xa"], r["ea"]) if side == "a" else (r["xb"], r["eb"])
        if shuf_pool is not None:
            o = random.choice(shuf_pool); x, e = (o["xa"], o["ea"]) if random.random() < .5 else (o["xb"], o["eb"])
        v = std(x)
        if a.emb: v = np.concatenate([v, pca.transform(e[None])[0] / 3.0])
        return torch.tensor(v, dtype=torch.float32)
    din = len(SCAL) * 2 + (32 if a.emb else 0)
    def build(r, arm, proj, const, shuf_pool=None):
        tA = r["da"]; tB = r["db"]
        if arm in ("N1", "N3"):
            tA += " [features: " + text_of(r["xa"]) + "]"; tB += " [features: " + text_of(r["xb"]) + "]"
        pre = tok.apply_chat_template([{"role": "user", "content": "Which liquid electrolyte has the higher ionic conductivity?\nA) " + tA}], tokenize=False, add_generation_prompt=False)
        pre = pre[: pre.rfind("<|im_end|>")]
        mid = "\nB) " + tB; post = "\nAnswer with the letter.<|im_end|>\n<|im_start|>assistant\nAnswer: "
        ids = lambda s: torch.tensor(tok(s, add_special_tokens=False).input_ids)
        if arm in ("N0", "N1"): sa, sb = const[0], const[1]
        else:
            sa = proj(vec(r, "a", shuf_pool)[None])[0]; sb = proj(vec(r, "b", shuf_pool)[None])[0]
        sa = sa / (sa.pow(2).mean().sqrt() + 1e-6) * rms; sb = sb / (sb.pow(2).mean().sqrt() + 1e-6) * rms
        emb = torch.cat([E(ids(pre)), sa, E(ids(mid)), sb, E(ids(post))])[None]
        return emb
    for arm in a.arms.split(","):
        key = arm + ("+emb" if a.emb and arm in ("N2", "N3", "N2-shuf") else "")
        if key in res: continue
        torch.manual_seed(0); random.seed(0)
        proj = Proj(din, d); const = nn.Parameter(torch.randn(2, 4, d) * 0.02)
        params = [const] if arm in ("N0", "N1") else list(proj.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
        pool = tr if arm == "N2-shuf" else None
        t0 = time.time(); proj.train(); losses = []
        for i, r in enumerate(tr):
            emb = build(r, arm, proj, const, pool)
            lg = lm(inputs_embeds=emb).logits[0, -1, [idA, idB]]
            loss = nn.functional.cross_entropy(lg[None], torch.tensor([r["y"]]))
            (loss / 8).backward(); losses.append(loss.item())
            if (i + 1) % 8 == 0: opt.step(); opt.zero_grad()
            if (i + 1) % 200 == 0: print(arm, i + 1, np.mean(losses[-200:]), f"{time.time()-t0:.0f}s", flush=True)
        proj.eval(); corr = []
        with torch.no_grad():
            for r in te:
                lg = lm(inputs_embeds=build(r, arm, proj, const, te if arm == "N2-shuf" else None)).logits[0, -1, [idA, idB]]
                corr.append(int(int(lg.argmax()) == r["y"]))
        res[key] = corr; print(key, "test acc", np.mean(corr), flush=True)
        Path(a.out).write_text(json.dumps(res))
    # zero-shot reference with no training (text only)
    Path(a.out).write_text(json.dumps(res))

if __name__ == "__main__":
    main()

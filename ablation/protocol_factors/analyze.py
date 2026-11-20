"""Analysis: per-task accuracy with cluster bootstrap CIs, paired deltas, permutation p-values, Holm/BH,
Shapley over the 4 info layers, dependent-vs-independent splits, ranked decisive-factor tables."""
import json, glob, itertools, math, sys, numpy as np
from pathlib import Path
from collections import defaultdict
from paths import DATA, RESULTS
B = 2000; RNG = np.random.default_rng(0)
LAYER_LABEL = {"Q": "qm", "F": "fm", "S": "solvation", "M": "micro"}
LAYERS = ["Q", "F", "S", "M"]

def load_items():
    it = {}
    for f in glob.glob(str(DATA / "items_*.jsonl")):
        for l in open(f):
            r = json.loads(l); it[r["id"]] = r
    return it

def load_results(path):
    res = defaultdict(dict)
    for l in open(path):
        r = json.loads(l); res[r["cond"]][r["id"]] = r
    return res

def task_of(item):
    return {"error": "error-correction", "reasoning": "reasoning", "planning": "planning"}[item["task"]]

def boot_ci(vals, groups, stat=np.mean):
    vals = np.asarray(vals, float); groups = np.asarray(groups)
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    est = stat(vals); bs = []
    for _ in range(B):
        pick = RNG.choice(ug, len(ug)); ii = np.concatenate([idx[g] for g in pick])
        bs.append(stat(vals[ii]))
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def perm_p(diffs, groups, n=5000):
    """Cluster sign-flip permutation test for mean paired difference."""
    diffs = np.asarray(diffs, float); groups = np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    gs = np.bincount(inv, weights=diffs); obs = abs(gs.sum())
    if obs == 0: return 1.0
    flips = RNG.choice([-1, 1], size=(n, len(ug)))
    return float((np.abs(flips @ gs) >= obs - 1e-12).mean())

def auroc(p, y):
    p = np.asarray(p); y = np.asarray(y).astype(bool)
    if y.all() or (~y).all(): return float("nan")
    pos, neg = p[y], p[~y]
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
    return adj

def bh(ps):
    ps = np.asarray(ps); m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 1
    for k in range(m - 1, -1, -1):
        i = order[k]; run = min(run, ps[i] * m / (k + 1)); adj[i] = run
    return adj

def per_condition_table(res, items, split="test"):
    rows = []
    for cond, d in res.items():
        by = defaultdict(list)
        for iid, r in d.items():
            it = items.get(iid)
            if not it or it.get("split", "test") != split: continue
            by[task_of(it)].append((r, it))
        for task, lst in by.items():
            acc = [float(r["correct"]) for r, _ in lst]; g = [it.get("group", it["id"]) for _, it in lst]
            e, lo, hi = boot_ci(acc, g)
            row = {"cond": cond, "task": task, "N": len(lst), "acc": e, "lo": lo, "hi": hi}
            if task == "error-correction":
                p = [r["p_error"] for r, _ in lst]; y = [it["is_error"] for _, it in lst]
                row["auroc"] = auroc(p, y)
                pred = np.array([r["pred_error"] for r, _ in lst]); yy = np.array(y)
                tp = (pred & yy).sum(); fp = (pred & ~yy).sum(); fn = (~pred & yy).sum()
                row["f1"] = float(2 * tp / max(1, 2 * tp + fp + fn))
                for v in ("c", "d"):
                    sel = [r["pred_error"] for r, it in lst if it.get("variant") == v]
                    if sel: row[f"fpr_{v}"] = float(np.mean(sel))
                loc = [float(r.get("correct_loc", r["correct"])) for r, _ in lst]
                row["acc_loc"] = float(np.mean(loc))
            rows.append(row)
    return rows

def paired(res, items, c1, c0, filt=lambda it: True, split="test", metric="correct"):
    if c1 not in res or c0 not in res: return None
    ids = [i for i in res[c1] if i in res[c0] and i in items and items[i].get("split", "test") == split and filt(items[i])]
    if not ids: return None
    d = [float(res[c1][i][metric]) - float(res[c0][i][metric]) for i in ids]
    g = [items[i].get("group", i) for i in ids]
    e, lo, hi = boot_ci(d, g)
    return {"delta": e, "lo": lo, "hi": hi, "N": len(ids), "p": perm_p(d, g)}

def cond_name(sub):
    return "B" if not sub else "+" + "".join(l for l in LAYERS if l in sub)

def shapley(res, items, filt=lambda it: True, split="test"):
    """Exact Shapley over 4 layers if all 16 cells exist; otherwise endpoint approximation (alone + LOO)."""
    cells = {frozenset(s): cond_name(s) for r in range(5) for s in itertools.combinations(LAYERS, r)}
    have = all(c in res for c in cells.values())
    ids = [i for i in res.get("B", {}) if i in items and items[i].get("split", "test") == split and filt(items[i])
           and all(i in res.get(c, {}) for c in cells.values() if c in res)]
    out = {}
    for L in LAYERS:
        contrib = np.zeros(len(ids)); wsum = 0
        others = [x for x in LAYERS if x != L]
        for r in range(4):
            for S in itertools.combinations(others, r):
                c0, c1 = cells[frozenset(S)], cells[frozenset(S) | {L}]
                if c0 not in res or c1 not in res: continue
                w = math.factorial(r) * math.factorial(3 - r) / math.factorial(4)
                contrib += w * np.array([float(res[c1][i]["correct"]) - float(res[c0][i]["correct"]) for i in ids]); wsum += w
        if wsum == 0 or not ids: continue
        contrib /= wsum
        g = [items[i].get("group", i) for i in ids]
        e, lo, hi = boot_ci(contrib, g)
        out[L] = {"shapley": e, "lo": lo, "hi": hi, "N": len(ids), "p": perm_p(contrib, g), "exact": have}
    return out

def diagonal(res, items, split="test"):
    """Pooled difference-in-differences: gain of layer L (alone, over B) on items labeled with L's domain minus its gain on other dependent items."""
    vals, groups = [], []
    for L in LAYERS:
        c = cond_name({L})
        if c not in res or "B" not in res: continue
        for i in res[c]:
            it = items.get(i)
            if not it or i not in res["B"] or it.get("split", "test") != split or it.get("bench") != "own": continue
            if it["label"] in ("none", "formal"): continue
            dlt = float(res[c][i]["correct"]) - float(res["B"][i]["correct"])
            vals.append((dlt, it["label"] == LAYER_LABEL[L])); groups.append(it["group"])
    if not vals: return None
    v = np.array([x for x, _ in vals]); m = np.array([y for _, y in vals]); g = np.array(groups)
    def stat_idx(ii): return v[ii][m[ii]].mean() - v[ii][~m[ii]].mean() if m[ii].any() and (~m[ii]).any() else 0
    ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}
    est = stat_idx(np.arange(len(v))); bs = []
    for _ in range(B):
        ii = np.concatenate([idx[k] for k in RNG.choice(ug, len(ug))]); bs.append(stat_idx(ii))
    return {"diag": est, "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5), "N": len(v),
            "p_two_sided_boot": float(2 * min((np.array(bs) <= 0).mean(), (np.array(bs) >= 0).mean()))}

def fmt_ci(d, k="delta"):
    return f"{d[k]*100:+.1f} [{d['lo']*100:+.1f}, {d['hi']*100:+.1f}]"

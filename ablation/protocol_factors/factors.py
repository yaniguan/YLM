"""Factorial effects with item-level bootstrap CIs.
A 'cell' is one setting of the factors (e.g. layers=QS, modality=text, probe=knn). Each cell has a per-item score
(0/1, or a mean over seeds). Main effect of a binary factor = mean over items of
[mean score over cells with the factor on] - [mean over matched cells with it off].
Two-way interaction of A and B = mean over items of (A on,B on - A off,B on) - (A on,B off - A off,B off)."""
import itertools, numpy as np

def _matrix(cells, ids):
    return {k: np.array([v[i] for i in ids], float) for k, v in cells.items()}

def effect(cells, key_fn, factor, on, off, ids, other=None):
    """cells: {cell_key: {item: score}}; key_fn(cell_key) -> dict of factor levels.
    Paired over cells that differ only in `factor`. Returns per-item effect vector."""
    M = _matrix(cells, ids); diffs = []
    lv = {k: key_fn(k) for k in cells}
    for k, f in lv.items():
        if f.get(factor) != on: continue
        if other and any(f.get(a) != b for a, b in other.items()): continue
        twin = [k2 for k2, f2 in lv.items() if f2.get(factor) == off and
                all(f2.get(a) == b for a, b in f.items() if a != factor)]
        if twin: diffs.append(M[k] - M[twin[0]])
    return np.mean(diffs, 0) if diffs else None

def interaction(cells, key_fn, fa, a_on, a_off, fb, b_on, b_off, ids):
    e1 = effect(cells, key_fn, fa, a_on, a_off, ids, other={fb: b_on})
    e0 = effect(cells, key_fn, fa, a_on, a_off, ids, other={fb: b_off})
    return None if e1 is None or e0 is None else e1 - e0

def boot(vec, groups=None, B=2000, seed=0):
    rng = np.random.default_rng(seed); vec = np.asarray(vec, float)
    groups = np.arange(len(vec)) if groups is None else np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=vec); cnt = np.bincount(inv)
    bs = []
    for _ in range(B):
        pick = rng.integers(0, len(ug), len(ug)); bs.append(sums[pick].sum() / cnt[pick].sum())
    flips = rng.choice([-1, 1], size=(5000, len(ug))); obs = abs(sums.sum())
    p = float((np.abs(flips @ sums) >= obs - 1e-12).mean()) if obs > 0 else 1.0
    return float(vec.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), p

def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj

def layer_key(name, layers="QFSM"):
    """'none' / 'B' / '+QS' / 'QS' -> {'Q': 1, 'F': 0, ...}"""
    s = "" if name in ("none", "B") else name.lstrip("+")
    return {L: int(L in s) for L in layers}

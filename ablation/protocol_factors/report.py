"""Build the result tables (markdown + JSON) from whatever result files exist. Cells without results are 'not run'."""
import json, glob, itertools, numpy as np
from pathlib import Path
from collections import defaultdict
from paths import DATA, RESULTS
import factors as F

LAYERS = "QFSM"
TASKS = {"error": "error detection", "reasoning": "protocol QA", "planning": "ordering/planning"}

def load_items():
    it = {}
    for f in glob.glob(str(DATA / "items_*.jsonl")):
        for l in open(f):
            r = json.loads(l); it[r["id"]] = r
    return it

def load_reader(pattern):
    res = defaultdict(dict)
    for f in glob.glob(str(RESULTS / Path(pattern).name) if pattern.startswith("results/") else pattern):
        for l in open(f):
            r = json.loads(l); res[r["cond"]][r["id"]] = float(r["correct"])
    return res

def ci(v, g):
    e, lo, hi, p = F.boot(v, g); return {"est": e, "lo": lo, "hi": hi, "p": p, "N": len(v)}

def pct(d, signed=False):
    if d is None: return "not run"
    f = "{:+.1f}" if signed else "{:.1f}"
    return (f + " [" + f + ", " + f + "]").format(100 * d["est"], 100 * d["lo"], 100 * d["hi"])

def own_tables(reader, res, items, out):
    """Accuracy per condition x task, and paired contrasts."""
    rows = []
    for cond, d in sorted(res.items()):
        for task in TASKS:
            ids = [i for i in d if items[i]["task"] == task and items[i]["split"] == "test"]
            if ids: rows.append({"reader": reader, "cond": cond, "task": TASKS[task],
                                 **ci([d[i] for i in ids], [items[i]["group"] for i in ids])})
    out["own_acc"] += rows
    def contrast(name, c1, c0, filt):
        for task in list(TASKS) + ["all"]:
            if c1 not in res or c0 not in res: continue
            ids = [i for i in res[c1] if i in res[c0] and filt(items[i]) and (task == "all" or items[i]["task"] == task)]
            if len(ids) < 5: continue
            v = [res[c1][i] - res[c0][i] for i in ids]
            out["contrasts"].append({"reader": reader, "factor": name, "task": TASKS.get(task, "all"),
                                     **ci(v, [items[i]["group"] for i in ids])})
    dep = lambda it: it["label"] not in ("none", "formal")
    contrast("all four layers (Q+F+S+M) vs none, dependent items", "+QFSM", "B", dep)
    contrast("all four layers vs none, layer-independent items", "+QFSM", "B", lambda it: it["label"] == "none")
    contrast("true layers vs shuffled layers (same length, wrong molecule)", "+QFSM", "ALL-shuf", dep)
    contrast("all layers vs length-matched placeholder facts", "+QFSM", "ALL-plac", dep)
    contrast("molecule identity (SMILES) vs no identity", "B", "A", lambda it: True)
    for L in LAYERS:
        name = {"Q": "QM descriptors (xTB)", "F": "mol-FM (MoLFormer heads)", "S": "solvation", "M": "microenvironment"}[L]
        contrast(f"{name} alone vs none, on its own label", "+" + L, "B",
                 lambda it, L=L: it["label"] == {"Q": "qm", "F": "fm", "S": "solvation", "M": "micro"}[L])
    formal = lambda it: it.get("lablean_expressible")
    contrast("structured form (R2) vs text", "R2", "B", lambda it: True)
    contrast("LabLean code (R3) vs text, LabLean-expressible items", "R3", "B", formal)
    contrast("LabLean + checker verdict (R4) vs text, expressible items", "R4", "B", formal)
    contrast("checker verdict (R4) vs LabLean code alone (R3)", "R4", "R3", formal)
    # 16-cell factorial over layers (only if all cells exist)
    cells = {c: d for c, d in res.items() if c == "B" or (c.startswith("+") and set(c[1:]) <= set(LAYERS))}
    if len(cells) == 16:
        ids = sorted(set.intersection(*[set(d) for d in cells.values()]))
        kf = lambda k: F.layer_key(k)
        for L in LAYERS:
            v = F.effect(cells, kf, L, 1, 0, ids)
            out["factorial"].append({"reader": reader, "term": f"main {L}", **ci(v, [items[i]["group"] for i in ids])})
        for a, b in itertools.combinations(LAYERS, 2):
            v = F.interaction(cells, kf, a, 1, 0, b, 1, 0, ids)
            out["factorial"].append({"reader": reader, "term": f"{a} x {b}", **ci(v, [items[i]["group"] for i in ids])})

def calisol_probe(out):
    p = RESULTS / "probe_calisol.json"
    if not p.exists(): return
    R = json.loads(p.read_text()); ids = list(range(len(R["y"])))
    cells = defaultdict(list)
    for c in R["cells"]: cells[(c["layers"], c["modality"], c["probe"])].append(c["correct"])
    cells = {k: {i: float(np.mean([s[i] for s in v])) for i in ids} for k, v in cells.items()}
    kf = lambda k: {**F.layer_key(k[0]), "mod": k[1], "probe": k[2]}
    for k, d in sorted(cells.items()):
        out["calisol_cells"].append({"layers": k[0], "modality": k[1], "probe": k[2],
                                     **ci(list(d.values()), ids),
                                     "seed_sd": float(np.std([np.mean(s) for s in
                                        [c["correct"] for c in R["cells"] if (c["layers"], c["modality"], c["probe"]) == k]]))})
    terms = []
    for L in LAYERS: terms.append((f"layer {L} (main)", F.effect(cells, kf, L, 1, 0, ids)))
    terms.append(("native vs text (main)", F.effect(cells, kf, "mod", "native", "text", ids)))
    terms.append(("both vs text (main)", F.effect(cells, kf, "mod", "both", "text", ids)))
    terms.append(("both vs native (main)", F.effect(cells, kf, "mod", "both", "native", ids)))
    terms.append(("kNN vs logistic probe (main)", F.effect(cells, kf, "probe", "knn", "logreg", ids)))
    for L in LAYERS:
        terms.append((f"layer {L} x native-vs-text", F.interaction(cells, kf, L, 1, 0, "mod", "native", "text", ids)))
    for a, b in itertools.combinations(LAYERS, 2):
        terms.append((f"{a} x {b}", F.interaction(cells, kf, a, 1, 0, b, 1, 0, ids)))
    terms.append(("native-vs-text x kNN", F.interaction(cells, kf, "mod", "native", "text", "probe", "knn", "logreg", ids)))
    for name, v in terms:
        if v is not None: out["calisol_terms"].append({"term": name, **ci(v, ids)})


def _expected_layer_names():
    names = []
    for r in range(5):
        for sub in itertools.combinations(LAYERS, r):
            names.append("".join(sub) or "none")
    return names

def main():
    items = load_items()
    out = defaultdict(list)
    for reader, pat in [("Qwen2.5-0.5B", "results/qwen_own*.jsonl"), ("External LLM", "results/llm_own*.jsonl")]:
        res = load_reader(pat)
        # skip a reader with almost no scored items (an interrupted run)
        n = sum(len(d) for d in res.values())
        if n < 20:
            continue
        if res: own_tables(reader, res, items, out)
    calisol_probe(out)
    for name, pat in [("Qwen2.5-0.5B zero-shot", "results/qwen_calisol*.jsonl")]:
        res = load_reader(pat)
        n = sum(len(d) for d in res.values())
        if n < 20:
            continue
        for cond, d in sorted(res.items()):
            ids = [i for i in d if items.get(i, {}).get("split") == "test"]
            if ids: out["calisol_llm"].append({"reader": name, "cond": cond, **ci([d[i] for i in ids], ids)})
    for name in ["n2_calisol.json", "lean_detector.json"]:
        f = RESULTS / name
        if f.exists(): out[f.stem] = json.loads(f.read_text())
    if out.get("calisol_terms"):
        ps = [row["p"] for row in out["calisol_terms"]]
        adj = F.holm(ps)
        for row, a in zip(out["calisol_terms"], adj):
            row["p_holm"] = float(a)
    have = sorted({c["layers"] for c in out.get("calisol_cells", [])})
    out["meta"] = {
        "calisol_layer_subsets_run": have,
        "calisol_layer_subsets_expected": _expected_layer_names(),
        "calisol_layer_subsets_missing": [x for x in _expected_layer_names() if x not in have],
        "calisol_seeds": [0, 1, 2, 3, 4],
    }
    (RESULTS / "summary.json").write_text(json.dumps(dict(out), indent=1))
    return dict(out)

if __name__ == "__main__":
    o = main()
    for k, v in o.items():
        print(k, len(v) if isinstance(v, list) else (list(v) if isinstance(v, dict) and k == "meta" else ""))

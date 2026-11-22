"""Tests for ablation/protocol_factors (CPU, no network, no model downloads)."""
import json, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "ablation" / "protocol_factors"
sys.path.insert(0, str(ROOT))

import factors as F  # noqa: E402


def test_factorial_main_effect_and_interaction():
    ids = list(range(200))
    def scores(q, s):
        n = 80 + 40 * q + 20 * s + 20 * q * s
        return {i: float(i < n) for i in ids}
    cells = {(q, s): scores(q, s) for q in (0, 1) for s in (0, 1)}
    kf = lambda k: {"Q": k[0], "S": k[1]}
    assert F.effect(cells, kf, "Q", 1, 0, ids).mean() == pytest.approx(0.25)
    assert F.interaction(cells, kf, "Q", 1, 0, "S", 1, 0, ids).mean() == pytest.approx(0.1)


def test_effect_pairs_only_matched_cells():
    ids = [0, 1]
    cells = {("QS",): {0: 1.0, 1: 1.0}, ("S",): {0: 0.0, 1: 1.0}, ("Q",): {0: 1.0, 1: 0.0}}
    kf = lambda k: F.layer_key(k[0])
    # only QS vs S is a matched pair for Q (Q vs none is missing)
    assert F.effect(cells, kf, "Q", 1, 0, ids).tolist() == [1.0, 0.0]


def test_boot_ci_and_signflip():
    v = np.r_[np.ones(30), np.zeros(10)]
    est, lo, hi, p = F.boot(v)
    assert est == pytest.approx(0.75) and lo <= est <= hi and p < 0.01
    est, lo, hi, p = F.boot(np.zeros(20))
    assert est == 0 and p == 1.0


def test_cluster_bootstrap_uses_groups():
    v = np.array([1, 1, 1, 1, 0, 0], float); g = ["a", "a", "a", "a", "b", "c"]
    est, lo, hi, _ = F.boot(v, g, B=500)
    assert est == pytest.approx(4 / 6) and lo <= est <= hi


def test_holm_monotone():
    adj = F.holm([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03) and adj[1] == pytest.approx(0.06) and adj[2] == pytest.approx(0.06)


def test_layer_key():
    assert F.layer_key("B") == {"Q": 0, "F": 0, "S": 0, "M": 0}
    assert F.layer_key("+QS") == {"Q": 1, "F": 0, "S": 1, "M": 0}


rdkit = pytest.importorskip("rdkit")


def test_conditions_cover_full_factorial_and_controls():
    import render
    c = render.factorial_conditions()
    cells = [k for k in c if k == "B" or (k.startswith("+") and set(k[1:]) <= set("QFSM"))]
    assert len(cells) == 16
    assert c["A"]["identity"] is False and c["ALL-shuf"]["shuffled"] and c["ALL-plac"]["placeholder"]


def test_shuffle_map_is_a_derangement():
    import render
    items = [{"molecules": [{"name": n, "smiles": s}]} for n, s in
             [("water", "O"), ("ethanol", "CCO"), ("acetone", "CC(C)=O"), ("benzene", "c1ccccc1")]]
    smap = render.make_smap(items)
    assert sorted(smap) == sorted(smap.values()) and all(k != v for k, v in smap.items())


def test_prompts_differ_only_in_facts_block():
    import adapters
    it = {"id": "x-q", "stem": "Which?", "choices": ["a", "b", "c", "d"], "answer_index": 2, "task": "reasoning"}
    p0 = adapters.prompt_mcq(it, "")
    p1 = adapters.prompt_mcq(it, "FACTS")
    assert p1.replace("Information:\nFACTS\n\n", "") == p0
    ch, a = adapters.permute(it)
    assert ch[a] == "c" and adapters.permute(it) == (ch, a)


def test_rep_text_marks_non_expressible_and_verdict(monkeypatch):
    import reps
    monkeypatch.setitem(reps._cache, "own", {"a": {"structured": "S", "lean": "L", "verdict": "REJECTED x"}})
    it = {"id": "a", "bench": "own", "task": "error", "protocol": "Step 1: x"}
    r = reps.rep_conditions()
    assert "Structured form:\nS" in reps.rep_text(it, r["R2"])
    assert "checker" not in reps.rep_text(it, r["R3"])
    assert reps.rep_text(it, r["R4"]).endswith("LabLean checker result: REJECTED x")
    it2 = dict(it, id="b")
    assert "not expressible" in reps.rep_text(it2, r["R3"])


def test_benchmark_twins_and_splits():
    rows = [json.loads(l) for l in open(ROOT / "data" / "items_own.jsonl")]
    by = {}
    for r in rows: by.setdefault(r["group"], []).append(r)
    for g, rs in by.items():
        v = {r["variant"]: r for r in rs}
        assert set(v) == {"c", "b", "d", "q"}
        assert v["b"]["is_error"] and not v["c"]["is_error"] and not v["d"]["is_error"]
        assert v["b"]["protocol"] != v["c"]["protocol"] != v["d"]["protocol"]
        assert len({r["split"] for r in rs}) == 1  # twins never cross splits
        assert 0 <= v["q"]["answer_index"] < 4
    from rdkit import Chem
    assert all(Chem.MolFromSmiles(m["smiles"]) for r in rows for m in r["molecules"])


def test_summary_numbers_match_result_files():
    s = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    assert len(s["calisol_cells"]) == 96
    assert len(s["calisol_terms"]) == 19
    for row in s["calisol_cells"]:
        assert 0 <= row["lo"] <= row["est"] <= row["hi"] <= 1


def test_shuffled_condition_does_not_leak_electrolyte_solvent_table():
    import render

    item = {"molecules": [{"name": "EC", "smiles": "O=C1OCCO1"}]}
    smap = {"O=C1OCCO1": "COC(=O)OC"}  # EC swapped with DMC
    true_s = render.electrolyte_lines(item, "S", shuffled=False)
    shuf_s = render.electrolyte_lines({"molecules": [{"name": "EC", "smiles": smap["O=C1OCCO1"]}]}, "S", shuffled=True)
    assert "1.9 cP" in true_s[0]
    assert "0.59 cP" in shuf_s[0] and "1.9 cP" not in shuf_s[0]

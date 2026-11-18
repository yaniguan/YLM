"""Tests for the lab world (LabLean copy), the SQL gate and the lab token factory."""
import os
import random

import pytest

from src import lab_world as W
from src import lab_token_factory as F
from src import sql_gate as SG

LABLEAN = os.environ.get("LABLEAN_DIR", os.path.join(os.path.dirname(__file__), "..", "lean"))


def world():
    cs = [W.container("BUF", 15000, 10000, "buffer"), W.container("MM", 15000, 500, "mm"),
          W.container("A1", 2000, 0, plate="P1"), W.container("A2", 2000, 1900, "water", plate="P1")]
    rs = [{"name": "buffer", "state": "roomTemp", "keep_cold": False},
          {"name": "mm", "state": "frozen", "keep_cold": True},
          {"name": "water", "state": "roomTemp", "keep_cold": False}]
    return W.make_state(cs, rs)


GOOD = [("thaw", "mm"), ("pick_tip",), ("transfer", "P200", "BUF", "A1", 500), ("drop_tip",),
        ("pick_tip",), ("transfer", "P20", "MM", "A1", 100), ("mix", "P200", "A1", 300, 3),
        ("drop_tip",), ("seal", "P1"), ("incubate", "P1", 37, 300)]
BROKEN = {
    "overflow A2 190.0 20.0 200.0": [("pick_tip",), ("transfer", "P200", "BUF", "A2", 200)],
    "notThawed mm": [("pick_tip",), ("transfer", "P20", "MM", "A1", 100)],
    "outOfRange P200 10.0": [("thaw", "mm"), ("pick_tip",), ("transfer", "P200", "MM", "A1", 100)],
    "noTip": [("transfer", "P200", "BUF", "A1", 500)],
    "contaminated MM": [("thaw", "mm"), ("pick_tip",), ("transfer", "P200", "BUF", "A1", 500),
                        ("transfer", "P20", "MM", "A1", 100)],
    "sealed A1": [("seal", "P1"), ("pick_tip",), ("transfer", "P200", "BUF", "A1", 500)],
    "badTemp 100": [("incubate", "P1", 100, 60)],
    "notEnough MM 5.0 100.0": [("thaw", "mm"), ("pick_tip",), ("transfer", "P200", "MM", "A1", 450),
                               ("drop_tip",), ("pick_tip",), ("transfer", "P1000", "MM", "A2", 1000)],
}


def test_mirror_good_protocol():
    r = W.check(world(), GOOD)
    assert r["ok"] and r["volumes"] == {"BUF": 9500, "MM": 400, "A1": 600, "A2": 1900}


@pytest.mark.parametrize("err", sorted(BROKEN))
def test_mirror_broken_protocols(err):
    r = W.check(world(), BROKEN[err])
    assert not r["ok"] and r["error"] == err and r["fail_step"] == len(BROKEN[err])


def test_lean_source_rejects_unsafe_ids():
    assert "TF.runIdx s0 1 p0" in W.lean_source([(world(), GOOD)])
    with pytest.raises(AssertionError):
        W.lean_source([(W.make_state([W.container('bad"id', 10)], []), [])])


@pytest.mark.skipif(not W.lean_available(LABLEAN), reason="needs a built LabLean (YLM PR #92) and lake")
def test_lean_agrees_with_mirror():
    rng = random.Random(7)
    agent = F.Agent(rng, p_blind=0.0)
    recs = []
    for level in (1, 2, 3, 4):
        recs += F.make_protocol_records(rng, 25, level, agent, p_err=0.3)
    cases = [(r["task"]["state"], r["steps"]) for r in recs] + \
        [(world(), GOOD)] + [(world(), s) for s in BROKEN.values()]
    assert W.lean_check(cases, LABLEAN) == [W.check(s, p) for s, p in cases]


def test_kept_records_have_true_claims():
    rng = random.Random(0)
    agent = F.Agent(rng, p_blind=1.0, learn=False)     # ignores every rule: many wrong claims
    recs = F.make_protocol_records(rng, 150, 2, agent, p_err=0.3)
    F.verify_protocols(recs, lablean_dir=None)
    kept = [r for r in recs if r["trust"] == "verified"]
    rejected = [r for r in recs if r["trust"] == "rejected"]
    assert kept and rejected
    for r in kept:
        truth = W.check(r["task"]["state"], r["steps"])
        c = F.claim_of(r["trace"])
        assert (c["ok"], c["fail_step"], c["error"]) == (truth["ok"], truth["fail_step"], truth["error"])
        assert [x[2] for x in r["trace"]] == [x[2] for x in W.trace(r["task"]["state"], r["steps"])]
        assert r["text"].isascii() and "\n" not in r["text"]


def test_careful_agent_without_slips_is_always_kept():
    rng = random.Random(1)
    recs = F.make_protocol_records(rng, 100, 3, F.Agent(rng, p_blind=0.0), p_err=0.3, p_slip=0.0)
    F.verify_protocols(recs, lablean_dir=None)
    assert all(r["trust"] == "verified" for r in recs)


def test_sql_gate_layers():
    con = SG.build_db(0)
    gold = "SELECT COUNT(*) FROM experiments WHERE status = 'done'"
    assert SG.gate(con, gold, gold)[0] is None
    assert SG.gate(con, "SELECT COUNT(* FROM experiments", gold)[0] == 0
    assert SG.gate(con, "DELETE FROM experiments", gold)[0] == 1
    assert SG.gate(con, "SELECT 1; DROP TABLE samples", gold)[0] == 1
    assert SG.gate(con, "SELECT 1;;", gold)[0] == 1
    assert SG.gate(con, gold + " ;  ", gold)[0] is None              # one trailing ';' is fine
    assert not SG.more_than_one_statement("SELECT ';' FROM experiments")   # ';' inside a string
    assert SG.gate(con, "SELECT COUNT(*) FROM experiments WHERE colour = 'x'", gold)[0] == 2
    assert SG.gate(con, "SELECT COUNT(*) FROM experiments", gold)[0] == 3
    assert con.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 30   # nothing deleted


def test_sql_second_database_catches_lucky_match():
    """Wrong SQL that matches gold on one database must still fail if a second one differs."""
    c = SG.lucky_pass_counts(range(40))
    still_pass = sum(SG.gate(SG.build_db(s), sql, gold, [SG.build_db(s + 7919)])[0] is None
                     for s in range(40) for _q, gold, ws in SG.questions() for _l, sql in ws)
    # measured for seeds 0-39: 32 wrong variants x 40 databases; 40 pass with one db, 4 with two
    assert c == {"wrong_queries": 1280, "pass_one_db": 40, "pass_two_db": 4}
    assert still_pass == 4


def test_factory_loop_provenance_and_dedup():
    kept, rejected, stats = F.run_factory(seed=3, rounds=3, per_round=60, sql_per_round=20,
                                          lablean_dir=None, checker_versions={"mirror": "x"})
    assert kept and rejected
    ids = [x["provenance"]["text_sha256"] for x in kept]
    assert len(ids) == len(set(ids))
    for x in kept:
        p = x["provenance"]
        assert p["trust"] == "verified" and p["first_fail"] is None
        assert p["source"] == "agent" and p["data"] == "synthetic" and p["checkers"] == {"mirror": "x"}
        assert "fail" not in p["verdicts"].values()
    for x in rejected:
        assert x["provenance"]["trust"] == "rejected" and x["provenance"]["first_fail"]
    fr = [r["frontier"] for r in stats["rounds"]]
    assert fr == sorted(fr) and fr[0] == 1


def test_dedup_across_seeds():
    seen, kept = set(), []
    for seed in (0, 1):
        k, _r, _s = F.run_factory(seed=seed, rounds=2, per_round=40, sql_per_round=30,
                                  lablean_dir=None, seen=seen)
        kept += k
    ids = [x["provenance"]["text_sha256"] for x in kept]
    assert len(ids) == len(set(ids)) == len(seen)

"""Small, fast tests for the token-quality test (no Lean, no web corpus needed)."""
import importlib.util
import os
import random

import numpy as np
import pytest
import torch
import torch.nn.functional as Fn

from src import lab_token_factory as F
from src import lab_world as W
from src import token_quality as TQ

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_script():
    spec = importlib.util.spec_from_file_location("tq_script", os.path.join(ROOT, "scripts", "42_token_quality.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def world():
    """Agent records, oracle docs and an eval v2 built like the script, at a small size."""
    recs, _ = TQ.agent_records(seed=1, n_tasks=200, lablean_dir=None)
    exclude = {TQ.protocol_hash(r["task"]["state"], r["steps"]) for r in recs}
    train = set(exclude)
    oracle = TQ.oracle_docs(200, 300, None, exclude)
    before = set(exclude)
    stats = {}
    items = TQ.build_eval(5, 160, None, exclude, stats=stats)
    return {"recs": recs, "train": train, "oracle": oracle, "items": items, "stats": stats,
            "eval_hashes": exclude - before, "before": before}


def test_ascii_and_vocab():
    t = TQ.to_ascii("Add 10 µL at 37 °C\tthen mix")
    assert t.isascii() and "uL" in t and "\t" not in t
    assert TQ.encode(t) and max(TQ.encode(t)) < len(TQ.VOCAB)


def test_stream_is_exact_budget():
    rng = random.Random(0)
    docs = ["abc" * 50] * 20
    assert len(TQ.stream(docs, 1000, rng)) == 1000


def test_stream_budget_edge_case():
    # 4 + 1 + 4 + 1 = 10 = budget after two documents, but their joined text has only 9
    # characters; the third document must be used instead of raising.
    docs = ["aaaa", "bbbb", "cccc"]
    text, ids = TQ.stream(docs, 10, random.Random(0), return_ids=True)
    assert len(text) == 10 and len(ids) == 3
    with pytest.raises(ValueError):
        TQ.stream(docs, 15, random.Random(0))


def test_stream_ids_match_shuffle():
    docs = [f"doc{i}-" + "x" * i for i in range(30)]
    text, ids = TQ.stream(docs, 200, random.Random(7), return_ids=True)
    shuffled = list(docs)
    random.Random(7).shuffle(shuffled)
    assert [docs[i] for i in ids] == shuffled[:len(ids)]
    assert text == "\n".join(shuffled[:len(ids)])[:200]


def test_script_uses_tq_stream(monkeypatch):
    S = load_script()
    calls = []
    real = TQ.stream

    def spy(*args, **kw):
        calls.append(args[1])
        return real(*args, **kw)

    monkeypatch.setattr(S.TQ, "stream", spy)
    pools = {"agent": [("abcdefghij" * 3, None)] * 50, "oracle": [("q? A: x", None)] * 400}
    texts = S.build_texts(pools, [300, 1000], ("agent", "oracle"))
    assert sorted(calls) == [300, 1000, 1000]                       # oracle only at the largest D
    assert {k: len(v[0]) for k, v in texts.items()} == {("agent", 300): 300, ("agent", 1000): 1000,
                                                         ("oracle", 1000): 1000}
    src = open(os.path.join(ROOT, "scripts", "42_token_quality.py")).read()
    assert "n += len(d) + 1" not in src                              # no inline copy of stream


def test_eval_items_are_held_out_and_well_formed(world):
    items, st = world["items"], world["stats"]
    assert len(world["eval_hashes"]) == 160 and not world["eval_hashes"] & world["before"]
    assert not world["eval_hashes"] & world["train"]
    assert st["protocols"] == 160 and st["verdict_blocks"] == 40
    kinds = {it["kind"] for it in items}
    assert kinds == {"verdict", "volume"}
    for it in items:
        assert len(it["options"]) == 4 and len(set(it["options"])) == 4
        assert TQ.no_prefix(it["options"])                            # no option is a prefix of another
        assert not any(o.endswith(".") for o in it["options"])        # no padding
        assert 0 <= it["answer"] < 4
        assert (it["q"] + "".join(it["options"])).isascii()


def test_eval_answers_are_checker_truth(monkeypatch):
    seen = {}
    real = TQ.check_cases

    def spy(cases, lablean_dir):
        truths = real(cases, lablean_dir)
        for (state, steps), t in zip(cases, truths):
            seen[TQ.protocol_hash(state, steps)[:16]] = (state, steps)
        return truths

    monkeypatch.setattr(TQ, "check_cases", spy)
    items = TQ.build_eval(11, 40, None, set())
    by_proto = {}
    for it in items:
        by_proto.setdefault(it["protocol"], []).append(it)
    for h, its in by_proto.items():
        truth = W.check(*seen[h])
        assert [it["kind"] for it in its] in (["verdict"], ["verdict", "volume"])
        assert its[0]["options"][its[0]["answer"]] == TQ.outcome_text(truth)
        if len(its) == 2:                                               # volume items: passing protocols only
            assert truth["ok"]
            target = its[1]["q"].split("final volume in ")[1].split("?")[0]
            assert its[1]["options"][its[1]["answer"]] == TQ.volume_text(truth["volumes"][target])


def test_blocks_are_exchangeable(world):
    blocks = {}
    for it in world["items"]:
        blocks.setdefault(it["block"], []).append(it)
    for b in blocks.values():
        assert len(b) == 4
        assert all(it["options"] == b[0]["options"] for it in b)        # one shared option list
        assert sorted(it["answer"] for it in b) == [0, 1, 2, 3]         # each option right once
        if b[0]["kind"] == "verdict":
            assert len({it["n_steps"] for it in b}) == 1
        else:
            assert len({it["level"] for it in b}) == 1


def test_question_blind_scorers_get_chance(world):
    items = world["items"]
    ans = np.array([it["answer"] for it in items])
    h = lambda o: int.from_bytes(o.encode()[:8].ljust(8, b"\0"), "little") % 97  # noqa: E731
    blind = {
        "first": lambda it: 0,
        "last": lambda it: 3,
        "longest": lambda it: int(np.argmax([len(o) for o in it["options"]])),
        "shortest": lambda it: int(np.argmin([len(o) for o in it["options"]])),
        "alphabetical": lambda it: it["options"].index(min(it["options"])),
        "pass_if_offered": lambda it: it["options"].index("pass") if "pass" in it["options"] else 0,
        "largest_number": lambda it: int(np.argmax([float(o.split()[0]) if it["kind"] == "volume"
                                                    else len(o) for o in it["options"]])),
        "hash": lambda it: int(np.argmax([h(o) for o in it["options"]])),
    }
    for name, f in blind.items():
        acc = np.mean([f(it) == a for it, a in zip(items, ans)])
        assert abs(acc - 0.25) <= 0.02, (name, acc)
    prior = TQ.prior_only_correct(items, world["oracle"])
    assert abs(prior.mean() - 0.25) <= 0.02


def test_no_truncation(world):
    for it in world["items"]:
        for o in it["options"]:
            assert len(it["q"]) + len(o) + 1 <= TQ.CTX + 1               # option + newline fits
    model = TQ.untrained(0)
    with pytest.raises(ValueError):
        TQ.option_logprob(model, ["x" * TQ.CTX], ["ab"])
    assert len(TQ.option_logprob(model, ["x" * (TQ.CTX - 1)], ["ab"])) == 1


def test_shuffled_questions_come_from_other_blocks(world):
    items = world["items"]
    qs = TQ.shuffled_questions(items, seed=0)
    by_q = {}
    for it in items:
        by_q.setdefault(it["q"], []).append(it)
    for it, q in zip(items, qs):
        donors = by_q[q]
        assert all(d["kind"] == it["kind"] and d["block"] != it["block"] for d in donors)


def test_verified_records_have_no_wrong_claims():
    recs, _ = TQ.agent_records(seed=2, n_tasks=60, lablean_dir=None)
    verified = [r for r in recs if r["trust"] == "verified"]
    assert verified and any(r["trust"] == "rejected" for r in recs)
    for r in verified:
        truth = W.check(r["task"]["state"], r["steps"])
        claim = F.claim_of(r["trace"])
        assert (claim["ok"], claim["fail_step"], claim["error"]) == (truth["ok"], truth["fail_step"], truth["error"])
        true_tr = W.trace(r["task"]["state"], r["steps"])
        assert [x[1] for x in r["trace"]] == [x[1] for x in true_tr]      # every step's outcome
        assert [x[2] for x in r["trace"]] == [x[2] for x in true_tr]      # every changed volume
        assert r["text"] == F.record_text(r["task"], r["steps"], true_tr)


def test_option_logprob_equals_manual_sum():
    torch.manual_seed(0)
    model = TQ.untrained(3)
    model.eval()
    prefixes, options = ["Q: x? A: ", "Q: longer question? A: "], ["fail at step 2: noTip", "pass"]
    got = TQ.option_logprob(model, prefixes, options)
    for p, o, g in zip(prefixes, options, got):
        ids = TQ.encode(p + o)
        with torch.no_grad():
            lp = Fn.log_softmax(model(torch.tensor([ids[:-1]])), -1)[0]
        manual = sum(lp[len(p) - 1 + j, ids[len(p) + j]].item() for j in range(len(o)))
        assert g == pytest.approx(manual, abs=1e-4)
    per_char = TQ.option_logprob(model, prefixes, options, per_char=True)
    assert [len(c) for c in per_char] == [len(o) for o in options]
    assert [c.sum() for c in per_char] == pytest.approx(got, abs=1e-4)


def test_tiny_training_and_scoring_run(world):
    torch.set_num_threads(1)
    text = ("Q: x? A: yes\n" * 400)
    model, losses = TQ.train(text, seed=0, steps=3, batch=2)
    assert len(losses) == 3 and np.isfinite(losses).all()
    items = world["items"][:24]
    lps = TQ.evaluate(model, items)
    assert set(lps) == set(TQ.LOGPROB_KEYS)
    assert all(v.shape == (len(items), 4) and np.isfinite(v).all() for v in lps.values())
    # the primary score is the plain sum over the unpadded option
    it = items[0]
    direct = TQ.option_logprob(model, [it["q"]] * 4, it["options"])
    assert lps["lp"][0] == pytest.approx(direct, abs=1e-4)
    corr = TQ.correct(lps, items)
    assert set(corr) == set(TQ.SCORES)
    assert (corr["primary"] == (lps["lp"].argmax(1) == [i["answer"] for i in items])).all()


def test_eval_v1_is_unchanged():
    """Eval v1 keeps its first-run options (padded, uniform distractors) for comparison."""
    recs, _ = TQ.agent_records(seed=1, n_tasks=40, lablean_dir=None)
    exclude = {TQ.protocol_hash(r["task"]["state"], r["steps"]) for r in recs}
    n_train = len(exclude)
    items = TQ.build_eval_v1(5, 30, None, exclude)
    assert len(exclude) == n_train + 30
    for it in items:
        assert len({len(o) for o in it["options"]}) == 1                # padded to equal lengths
    cal, raw = TQ.evaluate_v1(TQ.untrained(0), items)
    assert cal.shape == raw.shape == (len(items),)


def test_default_workers_fit_memory(monkeypatch):
    S = load_script()
    monkeypatch.setattr(S, "mem_available_gb", lambda: 2.5)
    monkeypatch.setattr(S.os, "cpu_count", lambda: 8)
    assert S.default_workers(24) == 2                                   # 2.5 GB / 0.93 GB per worker
    assert S.default_workers(1) == 1
    monkeypatch.setattr(S, "mem_available_gb", lambda: 0.5)
    assert S.default_workers(24) == 1
    assert S.neutral_path("/somewhere/else/WLP-Dataset") == "WLP-Dataset"
    assert S.neutral_path(os.path.join(ROOT, "runs", "x")) == os.path.join("runs", "x")

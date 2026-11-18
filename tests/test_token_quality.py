"""Small, fast tests for the token-quality test (no Lean, no web corpus needed)."""
import random

import numpy as np
import torch

from src import token_quality as TQ


def test_ascii_and_vocab():
    t = TQ.to_ascii("Add 10 µL at 37 °C\tthen mix")
    assert t.isascii() and "uL" in t and "\t" not in t
    assert TQ.encode(t) and max(TQ.encode(t)) < len(TQ.VOCAB)


def test_stream_is_exact_budget():
    rng = random.Random(0)
    docs = ["abc" * 50] * 20
    assert len(TQ.stream(docs, 1000, rng)) == 1000


def test_eval_items_are_held_out_and_well_formed():
    recs, _ = TQ.agent_records(seed=1, n_tasks=40, lablean_dir=None)
    train = {TQ.protocol_hash(r["task"]["state"], r["steps"]) for r in recs}
    exclude = set(train)
    items = TQ.build_eval(5, 30, None, exclude)
    assert len(exclude) == len(train) + 30
    kinds = {it["kind"] for it in items}
    assert "verdict" in kinds
    for it in items:
        assert len(it["options"]) == 4 and len(set(it["options"])) == 4
        assert len({len(o) for o in it["options"]}) == 1          # equal option lengths
        assert 0 <= it["answer"] < 4
        assert (it["q"] + it["options"][0]).isascii()


def test_verified_records_have_no_wrong_claims():
    recs, _ = TQ.agent_records(seed=2, n_tasks=60, lablean_dir=None)
    assert any(r["trust"] == "verified" for r in recs)
    assert any(r["trust"] == "rejected" for r in recs)


def test_tiny_training_and_scoring_run():
    torch.set_num_threads(1)
    text = ("Q: x? A: yes\n" * 400)
    model, losses = TQ.train(text, seed=0, steps=3, batch=2)
    assert len(losses) == 3 and np.isfinite(losses).all()
    items = [{"q": "Q: x? A: ", "options": ["yes", "no.", "may", "nah"], "answer": 0, "kind": "verdict"}]
    cal, raw = TQ.evaluate(model, items)
    assert cal.shape == (1,) and raw.shape == (1,)

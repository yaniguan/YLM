"""Fast offline tests for the lab/ scaffold (stdlib only)."""
import io
import json
from pathlib import Path

import pytest

from lab import distill, molecule_context as mc, protocol_qa as pqa

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "lab" / "fixtures" / "protocol_qa_tiny.jsonl"
OCSR = ROOT / "lab" / "fixtures" / "ocsr_predictions_tiny.csv"
OCSR_JSONL = ROOT / "lab" / "fixtures" / "ocsr_context_tiny.jsonl"
CFG = ROOT / "lab" / "configs" / "distill_protocol.json"


# ---------- protocol_qa ----------

def test_fixture_items_are_valid():
    items = pqa.load_jsonl(ITEMS)
    assert len(items) == 4
    for it in items:
        pqa.validate_item(it)


def test_gold_predictions_score_all_correct():
    items = pqa.load_jsonl(ITEMS)
    preds = [{"id": it["id"], "pred": it["answer"]} for it in items]
    res = pqa.score(items, preds)
    assert res["overall"]["correct"] == res["overall"]["n"] == 4
    assert res["by_task"]["order"]["pairwise_order_accuracy"] == 1.0


def test_mcq_letter_and_text_forms():
    choices = ["Hexane", "Water"]
    assert pqa.mcq_index("B", choices) == 1
    assert pqa.mcq_index("(b)", choices) == 1
    assert pqa.mcq_index(" water. ", choices) == 1
    assert pqa.mcq_index("C", choices) is None
    assert pqa.mcq_index("ethanol", choices) is None


def test_missing_and_wrong_predictions():
    items = pqa.load_jsonl(ITEMS)
    preds = [
        {"id": "toy-mcq-1", "pred": "A"},                       # wrong
        {"id": "toy-exact-1", "pred": " 0.1 "},                 # right after normalize
        {"id": "toy-order-1", "pred": ["s2", "s1", "s3", "s4"]},  # one swapped pair
        {"id": "not-an-item", "pred": "A"},
    ]
    res = pqa.score(items, preds)
    o = res["overall"]
    assert (o["n"], o["answered"], o["correct"]) == (4, 3, 1)
    assert res["by_task"]["order"]["pairwise_order_accuracy"] == pytest.approx(5 / 6)
    assert res["unknown_pred_ids"] == ["not-an-item"]


def test_pairwise_counts_missing_steps_as_wrong():
    assert pqa.pairwise_order_accuracy(["a", "b"], ["a", "b", "c"]) == pytest.approx(1 / 3)


def test_bad_items_rejected():
    with pytest.raises(ValueError):
        pqa.validate_item({"id": "x", "task": "mcq", "question": "q", "choices": ["a", "b"], "answer": "z"})
    with pytest.raises(ValueError):
        pqa.validate_item({"id": "x", "task": "order", "question": "q", "answer": ["a", "a"]})
    with pytest.raises(ValueError):
        pqa.validate_item({"id": "x", "task": "free", "question": "q", "answer": "a"})


def test_cli_scores_and_prompts(tmp_path, capsys):
    items = pqa.load_jsonl(ITEMS)
    preds = tmp_path / "p.jsonl"
    preds.write_text("\n".join(json.dumps({"id": it["id"], "pred": it["answer"]}) for it in items))
    out = tmp_path / "s.json"
    assert pqa.main(["--items", str(ITEMS), "--preds", str(preds), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["overall"]["accuracy"] == 1.0
    capsys.readouterr()
    assert pqa.main(["--items", str(ITEMS), "--print-prompts"]) == 0
    text = capsys.readouterr().out
    assert "(B) Toluene" in text and "toy-order-1" in text


# ---------- molecule_context ----------

def test_ocsr_rows_to_refs_and_flags():
    refs = mc.read_predictions_csv(OCSR, min_margin=0.5)
    assert [r.smiles for r in refs] == ["CCO", "F[C@H](Cl)Br", "C/C=C/C"]   # empty SMILES row skipped
    assert [r.ref for r in refs] == ["M1", "M2", "M3"]
    assert refs[0].flags == []
    assert refs[1].flags == ["low_margin", "stereo"]
    assert refs[2].flags == ["ocsr_rejected", "low_margin", "stereo"]


def test_drop_rejected_and_stream_input():
    text = OCSR.read_text()
    refs = mc.read_predictions_csv(io.StringIO(text), drop_rejected=True)
    assert [r.source_id for r in refs] == ["img_001", "img_002"]


def test_context_block_feeds_prompt():
    refs = mc.read_predictions_csv(OCSR, min_margin=0.5)
    block = mc.context_block(refs)
    assert block.startswith("Molecules (from OCSR")
    assert "- M2 (img_002): F[C@H](Cl)Br [low_margin, stereo]" in block
    item = pqa.load_jsonl(ITEMS)[0]
    prompt = pqa.build_prompt(item, block)
    assert prompt.index("Molecules") < prompt.index("Question:")
    assert mc.context_block([]) == ""


def test_ocsr_context_jsonl_keeps_only_trusted_rows(tmp_path):
    # Fixture = real output of ouroboros-ocsr `python -m lab.molecule_context tests/fixtures/lab_predictions.csv`.
    refs = mc.read_context_jsonl(OCSR_JSONL)
    assert [r.source_id for r in refs] == ["aspirin", "acetate_salt", "l_alanine"]
    assert refs[2].flags == ["stereo"] and refs[0].margin == 0.8
    with_review = mc.read_context_jsonl(OCSR_JSONL, only_usable=False)
    assert {r.source_id for r in with_review} == {"aspirin", "acetate_salt", "alanine_flat", "l_alanine", "r_group"}
    assert "review" in next(r for r in with_review if r.source_id == "r_group").flags
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x", "status": "ok", "smiles": "C"}) + "\n")
    with pytest.raises(ValueError, match="schema"):
        mc.read_context_jsonl(bad)


def test_prompt_has_eyes_and_physics_before_question():
    block = mc.context_block(mc.read_context_jsonl(OCSR_JSONL))
    physics = ["[physics] argon: 2 of 2 steps (0.1 ps requested), checkpoint step 10000."]
    prompt = pqa.build_prompt(pqa.load_jsonl(ITEMS)[0], block, physics)
    assert prompt.index("Molecules") < prompt.index("[physics]") < prompt.index("Question:")


def test_order_pred_as_comma_string():
    item = next(it for it in pqa.load_jsonl(ITEMS) if it["task"] == "order")
    res = pqa.score([item], [{"id": item["id"], "pred": "s1, s2, s3, s4"}])
    assert res["overall"]["correct"] == 1


# ---------- distill ----------

def test_shipped_config_dry_run(capsys):
    assert distill.main(["--config", str(CFG), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "1. SFT" in out and "2. DPO" in out and "3. GRPO" in out
    assert "metrics: not run yet" in out
    assert "BioProBench" in out and "WARNING" in out   # non-commercial license is flagged


def test_without_dry_run_refuses_to_train(capsys):
    assert distill.main(["--config", str(CFG)]) == 2


def _cfg():
    return json.loads(CFG.read_text())


def test_unknown_method_rejected():
    cfg = _cfg()
    cfg["stages"].append({"name": "x", "method": "other"})
    with pytest.raises(distill.ConfigError, match="unknown method"):
        distill.validate(cfg)


def test_stage_order_enforced():
    cfg = _cfg()
    cfg["stages"] = list(reversed(cfg["stages"]))
    with pytest.raises(distill.ConfigError, match="order|follow"):
        distill.validate(cfg)


def test_labbench_must_be_eval_only():
    cfg = _cfg()
    for d in cfg["data"]:
        if d["name"] == "LAB-Bench":
            d["use"] = "train_and_eval"
    with pytest.raises(distill.ConfigError, match="eval only"):
        distill.validate(cfg)


def test_bad_config_returns_error_code(tmp_path):
    cfg = _cfg()
    cfg["quantize"]["bits"] = 3
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    assert distill.main(["--config", str(p), "--dry-run"]) == 1


# ---------- commercial-safe protocol dataset (#68) ----------

def test_commercial_dataset_and_config_pass_strict_gate(capsys):
    comm_cfg = ROOT / "lab" / "configs" / "distill_commercial.json"
    comm_data = ROOT / "lab" / "data" / "commercial_protocols_v1.jsonl"
    comm_card = ROOT / "lab" / "data" / "DATA_CARD.md"
    assert comm_card.exists() and "CC-BY-NC-4.0" in comm_card.read_text(encoding="utf-8")

    items = pqa.load_jsonl(comm_data)
    assert len(items) >= 20
    tasks_seen = {it["task"] for it in items}
    assert tasks_seen == {"mcq", "exact", "order"}
    for it in items:
        pqa.validate_item(it)
        assert it.get("license") == "MIT"
        assert it.get("commercial_ok") is True

    preds = [{"id": it["id"], "pred": it["answer"]} for it in items]
    scored = pqa.score(items, preds)
    assert scored["overall"]["accuracy"] == 1.0
    assert scored["by_task"]["order"]["pairwise_order_accuracy"] == 1.0

    assert distill.main(["--config", str(comm_cfg), "--dry-run", "--commercial"]) == 0
    out = capsys.readouterr().out
    assert "mode=commercial-safe" in out
    assert "WARNING:" not in out


def test_commercial_gate_rejects_non_commercial_or_unlicensed_items(tmp_path):
    # Research config (contains BioProBench CC-BY-NC-4.0 + BioProt TBD) must fail under --commercial
    assert distill.main(["--config", str(CFG), "--dry-run", "--commercial"]) == 1

    # Local JSONL with a missing or CC-BY-NC item license must also be rejected
    bad_jsonl = tmp_path / "bad_items.jsonl"
    bad_jsonl.write_text(
        json.dumps({
            "id": "nc-1",
            "task": "exact",
            "question": "How many uL?",
            "answer": "10",
            "license": "CC-BY-NC-4.0",
        }) + "\n",
        encoding="utf-8",
    )
    comm_cfg = json.loads((ROOT / "lab" / "configs" / "distill_commercial.json").read_text(encoding="utf-8"))
    comm_cfg["data"][0]["path"] = str(bad_jsonl)
    with pytest.raises(distill.ConfigError, match="not commercial-safe"):
        distill.validate(comm_cfg, commercial=True)

    missing_lic = tmp_path / "missing_lic.jsonl"
    missing_lic.write_text(
        json.dumps({"id": "nl-1", "task": "exact", "question": "How many uL?", "answer": "10"}) + "\n",
        encoding="utf-8",
    )
    comm_cfg["data"][0]["path"] = str(missing_lic)
    with pytest.raises(distill.ConfigError, match="missing 'license'"):
        distill.validate(comm_cfg, commercial=True)

import copy
import json
import os
from pathlib import Path

import pytest

from labledger import Ledger, LedgerError, SchemaError, validate_event
from labledger.demo import build
from labledger.export import audio_text, check_training_record, to_training
from labledger.jsonschema import json_schema
from labledger.schema import check_quantity

HERE = os.path.dirname(__file__)
EXAMPLE = os.path.join(HERE, "..", "examples", "demo_campaign")
KW = {"origin": "synthetic_demo", "license": "CC0-1.0", "agent": {"type": "human", "id": "p:1"},
      "recorded_utc": "2026-01-01T00:00:00Z"}


@pytest.fixture
def demo(tmp_path):
    return build(str(tmp_path / "demo"))


def test_demo_verifies(demo):
    assert demo.verify() == []
    assert len(demo.events) == 29 and len(demo.current()) == 24


def test_reload_from_disk(demo):
    again = Ledger(demo.root)
    assert again.verify() == [] and again.events == demo.events


def test_tamper_breaks_chain(demo):
    lines = Path(demo.path).read_text().splitlines()
    ev = json.loads(lines[5])
    ev["body"]["status"] = "failed" if ev["body"].get("status") == "ok" else "ok"
    lines[5] = json.dumps(ev)
    Path(demo.path).write_text("\n".join(lines) + "\n")
    assert any("event_hash" in p for p in Ledger(demo.root).verify())


def test_deleted_line_breaks_chain(demo):
    lines = Path(demo.path).read_text().splitlines()
    del lines[3]
    Path(demo.path).write_text("\n".join(lines) + "\n")
    assert Ledger(demo.root).verify()


def test_blob_checksum(demo):
    art = next(e for e in demo.current().values() if e["level"] == "artifact")
    with open(os.path.join(demo.root, art["body"]["uri"]), "ab") as f:
        f.write(b"x")
    assert any("checksum" in p for p in demo.verify())


def test_hierarchy_rules(tmp_path):
    L = Ledger(str(tmp_path))
    L.add("project", "prj:A", {"title": "t", "goal": "g"}, **KW)
    with pytest.raises(LedgerError):  # experiment must sit under a campaign
        L.add("experiment", "exp:A", {"title": "t", "protocol_id": "p", "status": "planned",
                                      "started_utc": "2026-01-01T00:00:00Z"}, parent_id="prj:A", **KW)
    with pytest.raises(LedgerError):  # unknown parent
        L.add("campaign", "cmp:A", {"title": "t", "hypothesis": "h"}, parent_id="prj:X", **KW)
    with pytest.raises(LedgerError):  # no double add
        L.add("project", "prj:A", {"title": "t", "goal": "g"}, **KW)
    assert len(L.events) == 1


def test_supersede_and_retract(demo):
    cur = demo.current()
    a2 = cur["obs:DEMO-1.4a"]["body"]["quantities"]["A273_well_A2"]["value"]
    assert a2 == 0.081  # corrected value is live
    first = next(e for e in demo.events if e["record_id"] == "obs:DEMO-1.4a")
    assert first["body"]["quantities"]["A273_well_A2"]["value"] == 0.083  # old kept
    assert "obs:DEMO-2.2c" not in cur
    with pytest.raises(LedgerError):  # stale supersede
        demo.append("supersede", "observation", "obs:DEMO-1.4a", cur["obs:DEMO-1.4a"]["body"],
                    parent_id="stp:DEMO-1.4", supersedes=first["seq"], reason="r", **KW)


@pytest.mark.parametrize("q,ok", [
    ({"value": 1.0, "unit": "uL"}, True),
    ({"value": 1.0, "unit": "microliter"}, False),
    ({"value": float("nan"), "unit": "uL"}, False),
    ({"value": 1.0, "unit": "uL", "uncertainty": {"kind": "std", "value": -1, "method": "m"}}, False),
    ({"value": 1.0, "unit": "uL", "uncertainty": {"kind": "interval", "low": 0.9, "high": 1.1,
                                                  "method": "m"}}, True),
    ({"value": 1.0, "unit": "uL", "uncertainty": {"kind": "std", "value": 0.1}}, False),
])
def test_quantity(q, ok):
    if ok:
        check_quantity(q, "q")
    else:
        with pytest.raises(SchemaError):
            check_quantity(q, "q")


def _event(demo, rid):
    return copy.deepcopy(demo.current()[rid])


def test_event_rules(demo):
    bad = _event(demo, "stp:DEMO-2.2")
    bad["body"]["failure_tags"] = []
    with pytest.raises(SchemaError):  # failed step needs a tag
        validate_event(bad)
    bad = _event(demo, "obs:DEMO-2.2a")
    bad["body"]["failure_tags"] = []
    with pytest.raises(SchemaError):  # audio above threshold needs acoustic_anomaly
        validate_event(bad)
    bad = _event(demo, "obs:DEMO-1.4a")
    bad["recorded_utc"] = "2026-01-05 09:30"
    with pytest.raises(SchemaError):
        validate_event(bad)
    bad = _event(demo, "art:DEMO-1.1a")
    bad["body"]["sha256"] = "abc"
    with pytest.raises(SchemaError):
        validate_event(bad)


def test_training_export(demo):
    assert to_training(demo) == []  # synthetic data is blocked by default
    recs = to_training(demo, allow_synthetic=True)
    assert len(recs) == 5
    for r in recs:
        check_training_record(r)
        assert r["schema"] == "coscientist.v1" and r["source"]["origin"] == "synthetic_demo"
        assert set(r["ledger"]["record_ids"]) <= set(demo.current())
    fail = next(r for r in recs if r["id"] == "exp:DEMO-2:failures")
    assert fail["messages"][1]["content"] == "Step 2: acoustic_anomaly, instrument_fault"
    assert "<audio:0>" in fail["messages"][0]["content"]
    assert fail["audios"][0]["schema"] == "lab-ears/audio-event/v1"
    assert "flagged" in audio_text(fail["audios"][0], 0)
    run1 = next(r for r in recs if r["id"] == "exp:DEMO-1:failures")
    assert "<img:0>" in run1["messages"][0]["content"] and run1["geoms"][0]["unit"] == "bohr"
    assert "duplicate entry" not in json.dumps(recs)  # retracted record never exported
    assert "0.083" not in json.dumps(recs)  # superseded value never exported
    sim = next(r for r in recs if r["id"] == "exp:DEMO-3:failures")
    assert sim["sims"][0]["system"] == "Ar" and sim["sims"][0]["tag"] == "EXAMPLE"


def test_export_gates(demo):
    assert all(not r["id"].startswith("exp:DEMO-2")
               for r in to_training(demo, allow_synthetic=True, sealed_ids=["obs:DEMO-2.2a"]))
    text = Path(demo.path).read_text()
    Path(demo.path).write_text(text + text.splitlines()[0] + "\n")
    with pytest.raises(LedgerError):
        to_training(Ledger(demo.root), allow_synthetic=True)


def test_compatible_with_recipe_schema(demo):
    schema = pytest.importorskip("recipes.coscientist_posttrain.schema")
    for r in to_training(demo, allow_synthetic=True):
        schema.validate(r)


def test_json_schema_matches_events(demo):
    jsonschema = pytest.importorskip("jsonschema")
    v = jsonschema.Draft202012Validator(json_schema())
    for ev in demo.events:
        v.validate(ev)
    bad = copy.deepcopy(demo.events[0])
    bad["level"] = "plate"
    assert not v.is_valid(bad)


def test_checked_in_files_are_current(tmp_path):
    saved = json.loads(Path(HERE, "..", "schema", "labledger.v1.schema.json").read_text())
    assert saved == json_schema()
    fresh = build(str(tmp_path / "d"))
    assert Path(EXAMPLE, "events.jsonl").read_text() == Path(fresh.path).read_text()
    assert Ledger(EXAMPLE).verify() == []


def test_cli(demo, tmp_path, capsys):
    from labledger.__main__ import main

    assert main(["verify", demo.root]) == 0
    out = tmp_path / "t.jsonl"
    assert main(["export", demo.root, str(out), "--allow-synthetic"]) == 0
    assert len(out.read_text().splitlines()) == 5

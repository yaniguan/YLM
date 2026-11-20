"""Build a small SYNTHETIC demo campaign. All values, ids, lots and serials are made up.

    python -m labledger.demo --out labledger/examples/demo_campaign
"""

import argparse
import json
import os
import shutil
import struct
import time
import zlib

from .export import check_training_record, to_training, write_jsonl
from .ledger import Ledger

ORIGIN, LICENSE = "synthetic_demo", "CC0-1.0"
ROBOT = {"type": "robot", "id": "robot:DEMO-ARM-1"}
HUMAN = {"type": "human", "id": "person:demo-operator"}
SW = [{"name": "labledger", "version": "1.0.0"}]


def png(w, h, rgb):
    """A flat-color PNG with only zlib and struct (no imaging library)."""
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def q(value, unit, std=None, method="instrument spec", kind="spec"):
    out = {"value": value, "unit": unit}
    if std is not None:
        out["uncertainty"] = {"kind": kind, "value": std, "method": method}
    return out


def build(out):
    if os.path.exists(out):
        shutil.rmtree(out)
    L = Ledger(out)
    t = iter(f"2026-01-05T{9 + i // 60:02d}:{i % 60:02d}:00Z" for i in range(600))
    kw = {"origin": ORIGIN, "license": LICENSE, "software": SW}

    def add(level, rid, body, parent=None, agent=ROBOT):
        return L.add(level, rid, body, parent_id=parent, agent=agent, recorded_utc=next(t), **kw)

    def blob(parent, rid, data, media_type, name):
        art = L.put_blob(data, media_type, name)
        add("artifact", rid, art, parent)

    add("project", "prj:DEMO", {"title": "SYNTHETIC DEMO: dye dilution reads",
                                "goal": "Show the ledger format end to end."}, agent=HUMAN)
    add("campaign", "cmp:DEMO-1", {"title": "Caffeine dilution series (synthetic)",
                                   "hypothesis": "Absorbance at 273 nm is linear in concentration."},
        "prj:DEMO", HUMAN)

    # ---- experiment 1: completed, mixed modalities -----------------------------------------
    e1 = "exp:DEMO-1"
    add("experiment", e1, {"title": "Dilution and plate read, run 1", "protocol_id": "proto:DIL-v1",
                           "status": "running", "started_utc": "2026-01-05T09:00:00Z"}, "cmp:DEMO-1")
    add("step", "stp:DEMO-1.1", {"index": 1, "action": "Thaw stock tube on ice", "status": "ok",
                                 "params": {"time": q(10, "min")}}, e1, HUMAN)
    add("observation", "obs:DEMO-1.1a", {
        "modality": "image", "instrument_id": "cam:DEMO-CAM-1", "t_utc": "2026-01-05T09:05:00Z",
        "detail": {"smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "ocsr_flag": 0.05,
                   "what": "label drawing on stock tube LOT-DEMO-001"}}, "stp:DEMO-1.1")
    blob("obs:DEMO-1.1a", "art:DEMO-1.1a", png(8, 8, (200, 210, 230)), "image/png", "tube.png")
    add("observation", "obs:DEMO-1.1b", {
        "modality": "geom", "t_utc": "2026-01-05T09:06:00Z", "instrument_id": "sw:demo-builder",
        "detail": {"Z": [8, 1, 1], "pos": [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
                   "unit": "angstrom", "what": "solvent molecule (water)"}}, "stp:DEMO-1.1")
    add("step", "stp:DEMO-1.2", {
        "index": 2, "action": "Transfer 20 uL stock into 180 uL buffer, 5 times (1:10 series)",
        "status": "ok", "instrument_id": "pip:DEMO-P200",
        "params": {"volume": q(20, "uL", 0.2), "dilution": q(10, "1")}}, e1)
    add("step", "stp:DEMO-1.3", {"index": 3, "action": "Mix by pipetting 5 times", "status": "ok",
                                 "instrument_id": "pip:DEMO-P200",
                                 "params": {"volume": q(100, "uL", 1.0)}}, e1)
    add("step", "stp:DEMO-1.4", {"index": 4, "action": "Read absorbance spectrum, 220-320 nm",
                                 "status": "ok", "instrument_id": "rdr:DEMO-READER-1",
                                 "params": {"path_length": q(0.5, "cm", 0.01)}}, e1)
    add("observation", "obs:DEMO-1.4a", {
        "modality": "reading", "instrument_id": "rdr:DEMO-READER-1",
        "t_utc": "2026-01-05T09:30:00Z",
        "quantities": {"A273_well_A1": q(0.812, "{absorbance}", 0.004, "replicate std, n=3", "std"),
                       "A273_well_A2": q(0.083, "{absorbance}", 0.003, "replicate std, n=3", "std")}},
        "stp:DEMO-1.4")
    spec = "nm,absorbance\n" + "\n".join(f"{x},{0.8 * 2.718 ** (-((x - 273) / 18) ** 2):.4f}"
                                          for x in range(220, 321, 5))
    add("observation", "obs:DEMO-1.4b", {
        "modality": "spectrum", "instrument_id": "rdr:DEMO-READER-1",
        "t_utc": "2026-01-05T09:31:00Z",
        "quantities": {"peak": q(273, "nm", 1, "instrument spec")},
        "detail": {"x_unit": "nm", "y_unit": "{absorbance}", "n_points": 21, "well": "A1"}},
        "stp:DEMO-1.4")
    blob("obs:DEMO-1.4b", "art:DEMO-1.4b", spec.encode(), "text/csv", "A1_spectrum.csv")
    add("observation", "obs:DEMO-1.s", {
        "modality": "series", "instrument_id": "env:DEMO-THERMO-1",
        "t_utc": "2026-01-05T09:00:00Z", "t_end_utc": "2026-01-05T09:40:00Z",
        "quantities": {"room_T_mean": q(21.4, "Cel", 0.3, "sensor spec")},
        "detail": {"x_unit": "s", "y_unit": "Cel", "n_points": 41}}, e1)
    # correction: the A2 reading was typed with the wrong value; supersede, do not edit
    old = L.current()["obs:DEMO-1.4a"]
    fixed = json.loads(json.dumps(old["body"]))
    fixed["quantities"]["A273_well_A2"]["value"] = 0.081
    L.append("supersede", "observation", "obs:DEMO-1.4a", fixed, parent_id="stp:DEMO-1.4",
             agent=HUMAN, recorded_utc=next(t), supersedes=old["seq"],
             reason="A2 value copied wrong from the reader export", **kw)
    done = dict(L.current()[e1]["body"], status="completed", ended_utc="2026-01-05T09:40:00Z")
    L.append("supersede", "experiment", e1, done, parent_id="cmp:DEMO-1", agent=ROBOT,
             recorded_utc=next(t), supersedes=L.current()[e1]["seq"], reason="run finished", **kw)

    # ---- experiment 2: failed on a noisy centrifuge ---------------------------------------
    e2 = "exp:DEMO-2"
    add("experiment", e2, {"title": "Dilution and plate read, run 2", "protocol_id": "proto:DIL-v1",
                           "status": "running", "started_utc": "2026-01-05T10:00:00Z"}, "cmp:DEMO-1")
    add("step", "stp:DEMO-2.1", {"index": 1, "action": "Thaw stock tube on ice", "status": "ok",
                                 "params": {"time": q(10, "min")}}, e2, HUMAN)
    add("step", "stp:DEMO-2.2", {
        "index": 2, "action": "Spin tubes to collect liquid", "status": "failed",
        "instrument_id": "cfg:DEMO-CENTRIFUGE-1", "failure_tags": ["instrument_fault"],
        "params": {"speed": q(2000, "x g", 50), "time": q(1, "min")}}, e2)
    audio = {"schema": "lab-ears/audio-event/v1", "protocol_id": "proto:DIL-v1", "step": 2,
             "instrument_id": "cfg:DEMO-CENTRIFUGE-1", "start_utc": "2026-01-05T10:12:00.000Z",
             "duration_s": 10.0, "detector": "demo-threshold", "score": 0.91, "threshold": 0.6,
             "sample_rate_hz": 16000, "mic": "contact", "speech_band_dropped": True,
             "raw_audio_kept": False, "notes": "synthetic score", "flagged": True}
    add("observation", "obs:DEMO-2.2a", {
        "modality": "audio", "instrument_id": "mic:DEMO-CONTACT-1",
        "t_utc": "2026-01-05T10:12:00Z", "t_end_utc": "2026-01-05T10:12:10Z",
        "failure_tags": ["acoustic_anomaly"], "detail": audio}, "stp:DEMO-2.2")
    blob("obs:DEMO-2.2a", "art:DEMO-2.2a", json.dumps({"n_mels": 4, "frames": [[0.1, 0.2, 0.9, 0.3]]})
         .encode(), "application/json", "logmel_features.json")
    add("observation", "obs:DEMO-2.2b", {
        "modality": "text", "t_utc": "2026-01-05T10:13:00Z", "failure_tags": ["instrument_fault"],
        "text": "Rotor knocked and stopped. Tubes looked unbalanced."}, "stp:DEMO-2.2", HUMAN)
    add("observation", "obs:DEMO-2.2c", {
        "modality": "text", "t_utc": "2026-01-05T10:13:30Z", "text": "duplicate entry"},
        "stp:DEMO-2.2", HUMAN)
    L.append("retract", "observation", "obs:DEMO-2.2c", parent_id="stp:DEMO-2.2", agent=HUMAN,
             recorded_utc=next(t), reason="duplicate of obs:DEMO-2.2b", **kw)
    add("step", "stp:DEMO-2.3", {"index": 3, "action": "Read absorbance spectrum, 220-320 nm",
                                 "status": "skipped", "instrument_id": "rdr:DEMO-READER-1"}, e2)
    failed = dict(L.current()[e2]["body"], status="failed", ended_utc="2026-01-05T10:20:00Z")
    L.append("supersede", "experiment", e2, failed, parent_id="cmp:DEMO-1", agent=ROBOT,
             recorded_utc=next(t), supersedes=L.current()[e2]["seq"],
             reason="aborted after centrifuge fault", **kw)

    # ---- experiment 3: a simulation reference run -----------------------------------------
    e3 = "exp:DEMO-3"
    add("experiment", e3, {"title": "Reference argon fluid run (synthetic numbers)",
                           "protocol_id": "proto:SIM-AR-v1", "status": "completed",
                           "started_utc": "2026-01-05T11:00:00Z",
                           "ended_utc": "2026-01-05T11:05:00Z"}, "cmp:DEMO-1")
    add("step", "stp:DEMO-3.1", {"index": 1, "action": "Run argon simulation at 120 K",
                                 "status": "ok", "instrument_id": "sw:wignerflow",
                                 "params": {"T": q(120, "K")}}, e3,
        {"type": "software", "id": "sw:wignerflow"})
    add("observation", "obs:DEMO-3.1a", {
        "modality": "sim", "t_utc": "2026-01-05T11:05:00Z",
        "detail": {"engine": "wignerflow", "system": "Ar", "values": {"T_K": 120.0}}},
        "stp:DEMO-3.1", {"type": "software", "id": "sw:wignerflow"})
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="labledger/examples/demo_campaign")
    a = ap.parse_args()
    t0 = time.perf_counter()
    L = build(a.out)
    t1 = time.perf_counter()
    problems = L.verify()
    t2 = time.perf_counter()
    recs = to_training(L, allow_synthetic=True)
    for r in recs:
        check_training_record(r)
    write_jsonl(recs, os.path.join(a.out, "train.jsonl"))
    print(json.dumps({"events": len(L.events), "live_records": len(L.current()),
                      "blobs": len(os.listdir(L.blob_dir)), "training_examples": len(recs),
                      "problems": problems, "build_ms": round(1000 * (t1 - t0), 1),
                      "verify_ms": round(1000 * (t2 - t1), 1)}, indent=1))


if __name__ == "__main__":
    main()

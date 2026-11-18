"""Representation conditions: R2 structured scientific form, R3 LabLean code, R4 R3 + checker verdict.
Converted texts are precomputed into data/reps_<bench>.json by convert_reps.py."""
import json
from pathlib import Path
from paths import DATA
_cache = {}

def rep_conditions():
    return {"R2": {"kind": "rep", "rep": "R2"}, "R3": {"kind": "rep", "rep": "R3"},
            "R4": {"kind": "rep", "rep": "R4"},
            "R4+ALL": {"kind": "rep", "rep": "R4", "with_layers": True, "layers": {"Q", "F", "S", "M"}, "identity": True}}

def _load(bench):
    if bench not in _cache:
        p = DATA / f"reps_{bench}.json"; _cache[bench] = json.loads(p.read_text()) if p.exists() else {}
    return _cache[bench]

def rep_text(item, cond):
    r = _load(item["bench"]).get(item["id"], {})
    base = item.get("protocol") if item["task"] == "error" else item.get("context_text", "")
    if cond["rep"] == "R2":
        s = r.get("structured")
        return (base + "\n\nStructured form:\n" + s) if s else base
    lean = r.get("lean")
    if not lean:
        return base + "\n\nLabLean form: not expressible in LabLean (outside the liquid-handling model)."
    s = base + "\n\nLabLean form:\n" + lean
    if cond["rep"] == "R4":
        s += "\n\nLabLean checker result: " + r.get("verdict", "not run")
    return s

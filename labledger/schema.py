"""Schema labledger/v1: one event = one JSON object. Checks need only the standard library.

An event adds, supersedes or retracts one record. A record sits at one level and points to
its parent. `body` holds the level-specific fields. See README.md for the full field list.
"""

import math
import re
from datetime import datetime

SCHEMA = "labledger/v1"

# level -> allowed parent levels (None = no parent)
LEVELS = {
    "project": (None,),
    "campaign": ("project",),
    "experiment": ("campaign",),
    "step": ("experiment",),
    # an observation belongs to the step that made it, or to the experiment for a
    # continuous stream (room temperature, a long audio clip) with a time window
    "observation": ("step", "experiment"),
    "artifact": ("observation",),
}
OPS = {"add", "supersede", "retract"}
ORIGINS = {"real", "simulated", "model_generated", "synthetic_demo"}
AGENT_TYPES = {"human", "robot", "instrument", "software", "model"}
EXPERIMENT_STATUS = {"planned", "running", "completed", "failed", "aborted"}
STEP_STATUS = {"planned", "ok", "deviated", "failed", "skipped"}
MODALITIES = {"text", "reading", "image", "audio", "spectrum", "sim", "geom", "series"}
FAILURE_TAGS = {
    "instrument_fault", "acoustic_anomaly", "out_of_range", "contamination",
    "pipetting_error", "volume_mismatch", "sample_lost", "timeout", "operator_error",
    "sim_disagreement", "checker_fail", "calibration_expired", "other",
}
ARTIFACT_ROLES = {"raw", "derived", "thumbnail"}
UNCERTAINTY_KINDS = {"std", "interval", "spec", "none"}
# small UCUM-style unit list; "1" means dimensionless
UNITS = {
    "1", "%", "nL", "uL", "mL", "L", "mg", "g", "kg", "nM", "uM", "mM", "M", "mol",
    "Cel", "K", "s", "min", "h", "rpm", "x g", "nm", "um", "mm", "cm", "Angstrom", "bohr",
    "{absorbance}", "{RFU}", "pH", "Pa", "kPa", "bar", "Hz", "kHz", "dB", "eV",
    "kcal/mol", "kJ/mol", "ppm", "m/z", "cm-1", "mS/cm", "V", "mA",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RECORD_ID = re.compile(r"^[a-z]+:[A-Za-z0-9._-]+$")
UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")


class SchemaError(ValueError):
    pass


def _check(ok, msg):
    if not ok:
        raise SchemaError(msg)


def check_utc(t, where):
    """Returns the parsed time, so callers can compare times as datetimes."""
    _check(isinstance(t, str) and UTC.match(t), f"{where}: time must be UTC like 2026-01-01T00:00:00Z")
    try:
        return datetime.fromisoformat(t.replace("Z", "+00:00"))
    except ValueError as e:
        raise SchemaError(f"{where}: bad time {t!r} ({e})") from None


def _order(b, start, end, w):
    """If both times are set, check them and that end is not before start."""
    if b.get(end) is None:
        if b.get(start) is not None:
            check_utc(b[start], f"{w}.{start}")
        return
    t1 = check_utc(b[end], f"{w}.{end}")
    if b.get(start) is not None:
        _check(t1 >= check_utc(b[start], f"{w}.{start}"), f"{w}: {end} is before {start}")


def _in(v, choices):
    """Membership test that is safe for unhashable values (lists, dicts)."""
    return isinstance(v, str) and v in choices


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_quantity(q, where):
    """{value, unit, uncertainty?}. value may be null only with a failure tag elsewhere."""
    _check(isinstance(q, dict), f"{where}: quantity must be an object")
    v = q.get("value")
    _check(v is None or _num(v), f"{where}: bad value {v!r}")
    _check(isinstance(q.get("unit"), str) and q["unit"] in UNITS, f"{where}: unknown unit {q.get('unit')!r}")
    u = q.get("uncertainty")
    if u is None:
        return
    _check(isinstance(u, dict), f"{where}: uncertainty must be an object")
    kind = u.get("kind")
    _check(isinstance(kind, str) and kind in UNCERTAINTY_KINDS,
           f"{where}: uncertainty kind must be in {sorted(UNCERTAINTY_KINDS)}")
    _check(isinstance(u.get("method"), str) and u["method"], f"{where}: uncertainty needs a method")
    if kind in ("std", "spec"):
        _check(_num(u.get("value")) and u["value"] >= 0, f"{where}: {kind} uncertainty needs value >= 0")
    if kind == "interval":
        _check(_num(u.get("low")) and _num(u.get("high")) and u["low"] <= u["high"],
               f"{where}: interval needs low <= high")


def _quantities(d, where):
    _check(isinstance(d, dict), f"{where}: must be an object of quantities")
    for k, q in d.items():
        check_quantity(q, f"{where}.{k}")


def _str(body, key, where, required=True):
    v = body.get(key)
    if v is None and not required:
        return
    _check(isinstance(v, str) and v.strip(), f"{where}: {key} must be a non-empty string")


def _body_project(b, w):
    _str(b, "title", w)
    _str(b, "goal", w)


def _body_campaign(b, w):
    _str(b, "title", w)
    _str(b, "hypothesis", w)


def _body_experiment(b, w):
    _str(b, "title", w)
    _str(b, "protocol_id", w)
    _check(_in(b.get("status"), EXPERIMENT_STATUS),
           f"{w}: status must be in {sorted(EXPERIMENT_STATUS)}")
    check_utc(b.get("started_utc"), f"{w}.started_utc")
    _order(b, "started_utc", "ended_utc", w)


def _body_step(b, w):
    _check(isinstance(b.get("index"), int) and not isinstance(b["index"], bool) and b["index"] >= 1,
           f"{w}: index must be an int >= 1")
    _str(b, "action", w)
    _check(_in(b.get("status"), STEP_STATUS),f"{w}: status must be in {sorted(STEP_STATUS)}")
    _str(b, "instrument_id", w, required=False)
    _quantities(b.get("params", {}), f"{w}.params")
    _order(b, "started_utc", "ended_utc", w)
    if b["status"] == "deviated":
        _str(b, "deviation", w)
    _failures(b, w)


def _failures(b, w):
    tags = b.get("failure_tags", [])
    _check(isinstance(tags, list) and all(_in(t, FAILURE_TAGS) for t in tags),
           f"{w}: failure_tags must be from {sorted(FAILURE_TAGS)}")
    if b.get("status") == "failed":
        _check(tags, f"{w}: a failed record needs at least one failure tag")


def _body_observation(b, w):
    m = b.get("modality")
    _check(_in(m, MODALITIES), f"{w}: modality must be in {sorted(MODALITIES)}")
    _str(b, "instrument_id", w, required=m not in ("text", "sim"))
    check_utc(b.get("t_utc"), f"{w}.t_utc")
    _order(b, "t_utc", "t_end_utc", w)
    _quantities(b.get("quantities", {}), f"{w}.quantities")
    _failures(b, w)
    d = b.get("detail", {})
    _check(isinstance(d, dict), f"{w}: detail must be an object")
    if m == "text":
        _str(b, "text", w)
    elif m == "reading":
        _check(b.get("quantities"), f"{w}: a reading needs quantities")
    elif m == "audio":
        # same fields as lab-ears audio-event/v1
        _check(d.get("schema") == "lab-ears/audio-event/v1", f"{w}: audio detail must be lab-ears/audio-event/v1")
        for k in ("detector", "score", "threshold", "duration_s", "sample_rate_hz"):
            _check(k in d, f"{w}: audio detail needs {k}")
        _check(_num(d["score"]) and _num(d["threshold"]), f"{w}: audio score and threshold must be numbers")
        # privacy: raw audio may be kept only once the speech band has been removed
        _check(not (d.get("raw_audio_kept") and not d.get("speech_band_dropped")),
               f"{w}: raw audio still contains the speech band")
        if d["score"] > d["threshold"]:
            _check("acoustic_anomaly" in b.get("failure_tags", []),
                   f"{w}: score above threshold needs the acoustic_anomaly tag")
    elif m == "sim":
        for k in ("engine", "system", "values"):
            _check(k in d, f"{w}: sim detail needs {k}")
    elif m == "geom":
        Z, pos = d.get("Z"), d.get("pos")
        _check(isinstance(Z, list) and isinstance(pos, list) and len(Z) == len(pos)
               and all(isinstance(p, list) and len(p) == 3 for p in pos),
               f"{w}: geom needs Z and pos (N x 3)")
        _check(_in(d.get("unit"), ("bohr", "angstrom")), f"{w}: geom unit must be bohr or angstrom")
    elif m in ("spectrum", "series"):
        _check(_in(d.get("x_unit"), UNITS) and _in(d.get("y_unit"), UNITS),
               f"{w}: {m} detail needs x_unit and y_unit")


def safe_uri(uri):
    """A relative path inside the ledger directory: no absolute path, no '..', no backslash."""
    if not isinstance(uri, str) or not uri or "\\" in uri or ":" in uri:
        return False
    parts = uri.split("/")
    return all(p not in ("", ".", "..") for p in parts)


def _body_artifact(b, w):
    _check(isinstance(b.get("sha256"), str) and SHA256.match(b["sha256"]), f"{w}: bad sha256")
    _check(isinstance(b.get("size_bytes"), int) and not isinstance(b["size_bytes"], bool)
           and b["size_bytes"] >= 0, f"{w}: bad size_bytes")
    _str(b, "media_type", w)
    _str(b, "uri", w)
    _check(_in(b.get("role"), ARTIFACT_ROLES), f"{w}: role must be in {sorted(ARTIFACT_ROLES)}")
    _check(safe_uri(b["uri"]), f"{w}: uri must be a relative path inside the ledger, without '..'")
    if b["role"] == "raw":
        _check(b["uri"] == f"blobs/{b['sha256']}", f"{w}: a raw artifact uri must be blobs/<sha256>")


BODIES = {
    "project": _body_project, "campaign": _body_campaign, "experiment": _body_experiment,
    "step": _body_step, "observation": _body_observation, "artifact": _body_artifact,
}


def validate_event(ev):
    """Check one event on its own (no ledger context). Raises SchemaError."""
    _check(isinstance(ev, dict), "event must be an object")
    rid = ev.get("record_id", "?")
    w = f"seq {ev.get('seq', '?')} {rid}"
    _check(ev.get("schema") == SCHEMA, f"{w}: schema must be {SCHEMA}")
    _check(isinstance(ev.get("seq"), int) and not isinstance(ev["seq"], bool) and ev["seq"] >= 0,
           f"{w}: seq must be an int >= 0")
    _check(_in(ev.get("op"), OPS), f"{w}: op must be in {sorted(OPS)}")
    _check(isinstance(rid, str) and RECORD_ID.match(rid), f"{w}: record_id must look like 'exp:E1'")
    level = ev.get("level")
    _check(_in(level, LEVELS), f"{w}: level must be in {list(LEVELS)}")
    parent = ev.get("parent_id")
    _check(parent is None or isinstance(parent, str), f"{w}: parent_id must be a string or null")
    _check((parent is None) == (LEVELS[level] == (None,)), f"{w}: {level} parent_id is wrong")
    check_utc(ev.get("recorded_utc"), f"{w}.recorded_utc")
    _check(_in(ev.get("origin"), ORIGINS), f"{w}: origin must be in {sorted(ORIGINS)}")
    _str(ev, "license", w)
    prov = ev.get("provenance") or {}
    _check(isinstance(prov, dict), f"{w}: provenance must be an object")
    agent = prov.get("agent") or {}
    _check(isinstance(agent, dict) and _in(agent.get("type"), AGENT_TYPES) and agent.get("id"),
           f"{w}: provenance.agent needs type in {sorted(AGENT_TYPES)} and id")
    for k in ("software", "derived_from"):
        _check(isinstance(prov.get(k, []), list), f"{w}: provenance.{k} must be a list")
    if ev["op"] == "supersede":
        _check(isinstance(ev.get("supersedes"), int), f"{w}: supersede needs the seq it replaces")
        _str(ev, "reason", w)
    for k in ("prev_hash", "event_hash"):
        if k in ev:
            _check(isinstance(ev[k], str) and SHA256.match(ev[k]), f"{w}: bad {k}")
    if ev["op"] == "retract":
        _str(ev, "reason", w)
        return
    body = ev.get("body") or {}
    _check(isinstance(body, dict), f"{w}: body must be an object")
    BODIES[level](body, f"{w}.body")

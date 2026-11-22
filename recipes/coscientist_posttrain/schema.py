"""JSONL record schema v1: validate one record, load a file, filter by license.

Placeholders <geom:i>, <img:i>, <sim:i>, <audio:i> in messages must use the indices 0..len-1
of the matching array. Geometry is stored in bohr (the YLM encoder unit); angstrom input is converted.
"""
import copy
import json
import os
import re

SCHEMA = "coscientist.v1"
BOHR_PER_ANGSTROM = 1.8897261246
TASKS = {"protocol_qa", "error_fix", "order", "sql", "lean", "geom_qa", "ocsr_check", "sim_qa"}
ROLES = {"system", "user", "assistant"}
VERIFIERS = {"protocol", "lean", "sql", "exact", "numeric", "none"}
TAGS = {"MEASURED", "TARGET", "EXAMPLE"}
SIM_TAGS = TAGS | {"SIMULATED", "MODEL_GENERATED"}
SIM_SYSTEMS = {"Ar"}  # public WignerFlow results are argon only
PLACEHOLDER = re.compile(r"<(geom|img|sim|audio):(\d+)>")
ARRAYS = {"geom": "geoms", "img": "images", "sim": "sims", "audio": "audios"}


class SchemaError(ValueError):
    pass


def _check(ok, msg):
    if not ok:
        raise SchemaError(msg)


def check_tag(block, where, allowed=TAGS):
    """Validate provenance tag; MEASURED requires a non-empty `source` path."""
    tag = block.get("tag")
    _check(tag in allowed, f"{where}: tag must be one of {sorted(allowed)}, got {tag!r}")
    _check(tag != "MEASURED" or block.get("source"), f"{where}: MEASURED needs a source path")


def validate(rec):
    """Return a checked copy of `rec` with every geometry in bohr. Raises SchemaError."""
    rec = copy.deepcopy(rec)
    rid = rec.get("id")
    _check(isinstance(rid, str) and bool(rid.strip()), "record needs a non-empty string id")
    _check(rec.get("schema") == SCHEMA, f"{rid}: schema must be {SCHEMA}")
    _check(rec.get("task") in TASKS, f"{rid}: unknown task {rec.get('task')!r}")
    msgs = rec.get("messages") or []
    _check(
        msgs and all(m.get("role") in ROLES and isinstance(m.get("content"), str) for m in msgs),
        f"{rid}: messages need role in {sorted(ROLES)} and str content",
    )
    for key in ARRAYS.values():
        rec.setdefault(key, [])
    used = {k: set() for k in ARRAYS}
    geom_counts: dict[int, int] = {}
    for m in msgs:
        for kind, i in PLACEHOLDER.findall(m["content"]):
            idx = int(i)
            used[kind].add(idx)
            if kind == "geom":
                geom_counts[idx] = geom_counts.get(idx, 0) + 1
    for kind, key in ARRAYS.items():
        _check(
            used[kind] == set(range(len(rec[key]))),
            f"{rid}: <{kind}:i> placeholders {sorted(used[kind])} do not match {len(rec[key])} {key}",
        )
    _check(
        all(c == 1 for c in geom_counts.values()),
        f"{rid}: each <geom:i> placeholder must appear exactly once across messages",
    )
    for g in rec["geoms"]:
        _check(len(g["Z"]) == len(g["pos"]) and all(len(p) == 3 for p in g["pos"]), f"{rid}: bad geometry")
        unit = g.get("unit")
        _check(unit in {"bohr", "angstrom"}, f"{rid}: geometry unit must be bohr or angstrom")
        if unit == "angstrom":
            g["pos"] = [[c * BOHR_PER_ANGSTROM for c in p] for p in g["pos"]]
            g["unit"] = "bohr"
    for s in rec["sims"]:
        _check(
            s.get("engine") == "wignerflow" and s.get("system") in SIM_SYSTEMS,
            f"{rid}: sims must be wignerflow with system in {sorted(SIM_SYSTEMS)}",
        )
        check_tag(s, f"{rid} sim", allowed=SIM_TAGS)
    _check(rec.get("verifier", {}).get("kind") in VERIFIERS, f"{rid}: unknown verifier kind")
    src = rec.get("source") or {}
    _check(
        isinstance(src.get("license"), str) and isinstance(src.get("commercial_ok"), bool),
        f"{rid}: source needs license (str) and commercial_ok (bool)",
    )
    return rec


def load_jsonl(path, commercial=False, max_ocsr_flag=0.5):
    """Validate every line. commercial=True drops commercial_ok=false records;
    records with any image VERDICT flag above max_ocsr_flag are dropped from training."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = validate(json.loads(line))
            if commercial and not rec["source"]["commercial_ok"]:
                continue
            if any(im.get("ocsr_flag", 0.0) > max_ocsr_flag for im in rec["images"]):
                continue
            out.append(rec)
    return out


def check_config_tags(cfg, root=".", inherited=None, where="config"):
    """Every number in a config must sit under a dict (or ancestor) with a valid `tag`.
    MEASURED needs a `source` file that exists (relative to `root`)."""
    if isinstance(cfg, dict):
        if "tag" in cfg:
            check_tag(cfg, where)
            if cfg["tag"] == "MEASURED":
                _check(os.path.exists(os.path.join(root, cfg["source"])), f"{where}: missing {cfg['source']}")
            inherited = cfg["tag"]
        for k, v in cfg.items():
            check_config_tags(v, root, inherited, f"{where}.{k}")
    elif isinstance(cfg, list):
        for i, v in enumerate(cfg):
            check_config_tags(v, root, inherited, f"{where}[{i}]")
    elif isinstance(cfg, (int, float)) and not isinstance(cfg, bool):
        _check(inherited is not None, f"{where}: number without a tag")

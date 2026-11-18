"""Export labledger/v1 as JSON Schema (draft 2020-12) for tools in other languages.

The Python checks in schema.py are the reference; this file mirrors their main rules.
    python -m labledger.jsonschema > labledger/schema/labledger.v1.schema.json
"""

import json

from . import schema as S

UTC = {"type": "string", "pattern": S.UTC.pattern}
NAME = {"type": "string", "minLength": 1}
QUANTITY = {
    "type": "object", "required": ["value", "unit"],
    "properties": {
        "value": {"type": ["number", "null"]},
        "unit": {"enum": sorted(S.UNITS)},
        "uncertainty": {
            "type": "object", "required": ["kind", "method"],
            "properties": {"kind": {"enum": sorted(S.UNCERTAINTY_KINDS)},
                           "value": {"type": "number", "minimum": 0},
                           "low": {"type": "number"}, "high": {"type": "number"},
                           "method": NAME}},
    },
}
QMAP = {"type": "object", "additionalProperties": {"$ref": "#/$defs/quantity"}}
TAGS = {"type": "array", "items": {"enum": sorted(S.FAILURE_TAGS)}}


def _obj(required, **props):
    return {"type": "object", "required": required, "properties": props}


BODIES = {
    "project": _obj(["title", "goal"], title=NAME, goal=NAME),
    "campaign": _obj(["title", "hypothesis"], title=NAME, hypothesis=NAME),
    "experiment": _obj(["title", "protocol_id", "status", "started_utc"], title=NAME,
                       protocol_id=NAME, status={"enum": sorted(S.EXPERIMENT_STATUS)},
                       started_utc=UTC, ended_utc=UTC),
    "step": _obj(["index", "action", "status"], index={"type": "integer", "minimum": 1},
                 action=NAME, status={"enum": sorted(S.STEP_STATUS)}, instrument_id=NAME,
                 params=QMAP, started_utc=UTC, ended_utc=UTC, deviation=NAME, failure_tags=TAGS),
    "observation": _obj(["modality", "t_utc"], modality={"enum": sorted(S.MODALITIES)},
                        instrument_id=NAME, t_utc=UTC, t_end_utc=UTC, quantities=QMAP,
                        text=NAME, failure_tags=TAGS, detail={"type": "object"}),
    "artifact": _obj(["sha256", "size_bytes", "media_type", "uri", "role"],
                     sha256={"type": "string", "pattern": S.SHA256.pattern},
                     size_bytes={"type": "integer", "minimum": 0}, media_type=NAME, uri=NAME,
                     filename_orig={"type": "string"}, role={"enum": sorted(S.ARTIFACT_ROLES)}),
}


def json_schema():
    hexs = {"type": "string", "pattern": S.SHA256.pattern}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/yaniguan/YLM/labledger/schema/labledger.v1.schema.json",
        "title": "labledger/v1 event",
        "type": "object",
        "required": ["schema", "seq", "op", "level", "record_id", "parent_id", "recorded_utc",
                     "origin", "license", "provenance", "body", "prev_hash", "event_hash"],
        "properties": {
            "schema": {"const": S.SCHEMA},
            "seq": {"type": "integer", "minimum": 0},
            "op": {"enum": sorted(S.OPS)},
            "level": {"enum": list(S.LEVELS)},
            "record_id": {"type": "string", "pattern": S.RECORD_ID.pattern},
            "parent_id": {"type": ["string", "null"]},
            "recorded_utc": UTC,
            "origin": {"enum": sorted(S.ORIGINS)},
            "license": NAME,
            "provenance": _obj(["agent"], agent=_obj(["type", "id"], type={"enum": sorted(S.AGENT_TYPES)},
                                                     id=NAME),
                               software={"type": "array"}, derived_from={"type": "array"}),
            "body": {"type": ["object", "null"]},
            "supersedes": {"type": "integer", "minimum": 0},
            "reason": NAME,
            "prev_hash": hexs,
            "event_hash": hexs,
        },
        "allOf": [
            {"if": {"properties": {"op": {"const": "supersede"}}},
             "then": {"required": ["supersedes", "reason"]}},
            {"if": {"properties": {"op": {"const": "retract"}}}, "then": {"required": ["reason"]}},
        ] + [
            {"if": {"properties": {"level": {"const": lv}, "op": {"enum": ["add", "supersede"]}}},
             "then": {"properties": {"body": {"$ref": f"#/$defs/{lv}"}}}}
            for lv in BODIES
        ],
        "$defs": dict(BODIES, quantity=QUANTITY),
    }


if __name__ == "__main__":
    print(json.dumps(json_schema(), indent=1, sort_keys=True))

"""Append-only ledger: events.jsonl (hash chain) + blobs/<sha256> (write-once raw files).

Nothing is ever edited. A correction is a `supersede` event with the full new body; a
withdrawal is a `retract` event. `current()` gives the latest live version of each record.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

from .schema import LEVELS, SCHEMA, SchemaError, validate_event

GENESIS = "0" * 64


class LedgerError(ValueError):
    pass


def canonical(obj):
    """Sorted keys, no spaces, UTF-8. NaN/Inf are rejected."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def event_hash(ev):
    body = {k: v for k, v in ev.items() if k != "event_hash"}
    return hashlib.sha256(canonical(body)).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Ledger:
    """One directory: events.jsonl, blobs/. Only `append` writes events."""

    def __init__(self, root):
        self.root = root
        self.path = os.path.join(root, "events.jsonl")
        self.blob_dir = os.path.join(root, "blobs")
        os.makedirs(self.blob_dir, exist_ok=True)
        self.events = self._read() if os.path.exists(self.path) else []
        self.latest = self._state()  # record_id -> latest event, kept up to date by append

    def _read(self):
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    # ---- writing -------------------------------------------------------------------------
    def append(self, op, level, record_id, body=None, *, parent_id=None, origin, license,
               agent, software=(), derived_from=(), recorded_utc=None, supersedes=None,
               reason=None):
        """Validate, chain and append one event. Returns the stored event."""
        ev = {
            "schema": SCHEMA, "seq": len(self.events), "op": op, "level": level,
            "record_id": record_id, "parent_id": parent_id,
            "recorded_utc": recorded_utc or now_utc(), "origin": origin, "license": license,
            "provenance": {"agent": dict(agent), "software": list(software),
                           "derived_from": list(derived_from)},
            "body": body if op != "retract" else None,
            "prev_hash": self.events[-1]["event_hash"] if self.events else GENESIS,
        }
        if supersedes is not None:
            ev["supersedes"] = supersedes
        if reason is not None:
            ev["reason"] = reason
        validate_event(ev)
        self._check_context(ev, self.latest)
        ev["event_hash"] = event_hash(ev)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(canonical(ev).decode("utf-8") + "\n")
        self.events.append(ev)
        self.latest[record_id] = ev
        return ev

    def add(self, level, record_id, body, **kw):
        return self.append("add", level, record_id, body, **kw)

    def put_blob(self, data, media_type, filename):
        """Store bytes once under their sha256. Returns an artifact body."""
        digest = hashlib.sha256(data).hexdigest()
        path = os.path.join(self.blob_dir, digest)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(data)
        return {"sha256": digest, "size_bytes": len(data), "media_type": media_type,
                "uri": f"blobs/{digest}", "filename_orig": filename, "role": "raw"}

    # ---- reading -------------------------------------------------------------------------
    def _state(self):
        """record_id -> latest event (any op)."""
        return {ev["record_id"]: ev for ev in self.events}

    @staticmethod
    def _check_context(ev, state):
        rid, prev = ev["record_id"], state.get(ev["record_id"])
        if ev["op"] == "add":
            if prev is not None:
                raise LedgerError(f"{rid}: already exists; use supersede")
        else:
            if prev is None or prev["op"] == "retract":
                raise LedgerError(f"{rid}: {ev['op']} of a missing or retracted record")
            if prev["level"] != ev["level"] or prev["parent_id"] != ev["parent_id"]:
                raise LedgerError(f"{rid}: level and parent cannot change")
            if ev["op"] == "supersede" and ev["supersedes"] != prev["seq"]:
                raise LedgerError(f"{rid}: supersedes must be the latest seq {prev['seq']}")
        if ev["parent_id"] is not None:
            parent = state.get(ev["parent_id"])
            if parent is None or parent["op"] == "retract":
                raise LedgerError(f"{rid}: parent {ev['parent_id']} is missing or retracted")
            if parent["level"] not in LEVELS[ev["level"]]:
                raise LedgerError(f"{rid}: a {ev['level']} cannot sit under a {parent['level']}")

    def current(self):
        """record_id -> live event (latest add/supersede, retracted records dropped)."""
        return {k: v for k, v in self.latest.items() if v["op"] != "retract"}

    def children(self, record_id, level=None):
        cur = self.current()
        out = [e for e in cur.values() if e["parent_id"] == record_id
               and (level is None or e["level"] == level)]
        return sorted(out, key=lambda e: (e["body"].get("index", 0), e["body"].get("t_utc", ""),
                                     e["record_id"]))

    # ---- checking ------------------------------------------------------------------------
    def verify(self, check_blobs=True):
        """Re-check every event, the hash chain, the tree and the blob checksums.
        Returns a list of problems (empty = clean)."""
        problems, prev, state = [], GENESIS, {}
        for i, ev in enumerate(self.events):
            try:
                validate_event(ev)
                if ev["seq"] != i:
                    raise LedgerError(f"seq {ev['seq']} at line {i}")
                if ev["prev_hash"] != prev:
                    raise LedgerError(f"seq {i}: prev_hash breaks the chain")
                if ev.get("event_hash") != event_hash(ev):
                    raise LedgerError(f"seq {i}: event_hash does not match content")
                self._check_context(ev, state)
            except (SchemaError, LedgerError) as e:
                problems.append(str(e))
            prev = ev.get("event_hash", "")
            state[ev.get("record_id")] = ev
        if check_blobs:
            for ev in self.current().values():
                if ev["level"] != "artifact":
                    continue
                b = ev["body"]
                path = os.path.join(self.root, b["uri"])
                if not os.path.exists(path):
                    problems.append(f"{ev['record_id']}: missing raw file {b['uri']}")
                elif sha256_file(path) != b["sha256"] or os.path.getsize(path) != b["size_bytes"]:
                    problems.append(f"{ev['record_id']}: checksum or size mismatch for {b['uri']}")
        return problems

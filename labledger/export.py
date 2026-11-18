"""Turn finished experiments into training JSONL (schema coscientist.v1, YLM PR #86).

Placeholders: <geom:i> -> `geoms`, <img:i> -> `images`, <sim:i> -> `sims`, <audio:i> -> `audios`.
Every example keeps `ledger` links (record ids + head hash) so it can be traced back to raw data.
Gates: the ledger must verify clean; retracted and superseded versions are never used;
synthetic_demo data is blocked unless allow_synthetic=True; sealed ids are never exported.
"""

import hashlib
import json
import random

from .ledger import LedgerError

BOHR_PER_ANGSTROM = 1.8897261246
COMMERCIAL_OK = {"own", "CC0-1.0", "CC-BY-4.0", "MIT", "Apache-2.0", "BSD-3-Clause"}
DONE = {"completed", "failed", "aborted"}


def _q(q):
    s = "?" if q["value"] is None else f"{q['value']:g}"
    u = q.get("uncertainty")
    if u and u["kind"] in ("std", "spec"):
        s += f" +/- {u['value']:g}"
    elif u and u["kind"] == "interval":
        s += f" [{u['low']:g}, {u['high']:g}]"
    return s if q["unit"] == "1" else f"{s} {q['unit']}"


def _qs(d):
    return ", ".join(f"{k}={_q(v)}" for k, v in d.items())


class _Media:
    """Collects multimodal blocks and returns their placeholders."""

    def __init__(self, synthetic):
        self.geoms, self.images, self.sims, self.audios = [], [], [], []
        self.tag = "EXAMPLE" if synthetic else "MEASURED"

    def add(self, ob, artifacts):
        b, d, rid = ob["body"], ob["body"].get("detail", {}), ob["record_id"]
        m = b["modality"]
        uri = artifacts[0]["body"]["uri"] if artifacts else None
        if m == "geom":
            k = 1.0 if d["unit"] == "bohr" else BOHR_PER_ANGSTROM
            self.geoms.append({"Z": d["Z"], "pos": [[c * k for c in p] for p in d["pos"]],
                               "unit": "bohr", "record_id": rid})
            return f"<geom:{len(self.geoms) - 1}>"
        if m == "image" and d.get("smiles"):
            self.images.append({"uri": uri, "smiles": d["smiles"],
                                "ocsr_flag": d.get("ocsr_flag", 0.0), "record_id": rid})
            return f"<img:{len(self.images) - 1}>"
        if m == "sim" and d["engine"] == "wignerflow" and d["system"] == "Ar":
            sim = {"engine": "wignerflow", "system": "Ar", "values": d["values"],
                   "tag": self.tag, "record_id": rid}
            if self.tag == "MEASURED":
                sim["source"] = uri or rid
            self.sims.append(sim)
            return f"<sim:{len(self.sims) - 1}>"
        if m == "audio":
            self.audios.append(dict(d, record_id=rid, features_path=uri))
            return f"<audio:{len(self.audios) - 1}>"
        # everything else is plain text
        if m == "text":
            return f'note: "{b["text"]}"'
        if m == "sim":
            return f"[sim {d['engine']} {d['system']}: " + ", ".join(
                f"{k}={v:g}" for k, v in d["values"].items()) + "]"
        if m in ("spectrum", "series"):
            return f"[{m} {d['x_unit']} vs {d['y_unit']}, {d.get('n_points', '?')} points" + (
                f"; {_qs(b['quantities'])}]" if b.get("quantities") else "]")
        if m == "image":
            return f"[image {b.get('instrument_id', '')}]"
        return _qs(b.get("quantities", {}))


def _render(ledger, exp, synthetic):
    media, used, lines = _Media(synthetic), [exp["record_id"]], []
    for st in ledger.children(exp["record_id"], "step"):
        b = st["body"]
        used.append(st["record_id"])
        line = f"Step {b['index']}: {b['action']}"
        if b.get("params"):
            line += f" ({_qs(b['params'])})"
        if b.get("instrument_id"):
            line += f" on {b['instrument_id']}"
        line += f". Status: {b['status']}."
        if b.get("deviation"):
            line += f" Deviation: {b['deviation']}."
        obs = []
        for ob in ledger.children(st["record_id"], "observation"):
            arts = ledger.children(ob["record_id"], "artifact")
            used += [ob["record_id"]] + [a["record_id"] for a in arts]
            obs.append(f"{ob['body']['modality']}: {media.add(ob, arts)}")
        if obs:
            line += " Observed: " + "; ".join(obs) + "."
        lines.append(line)
    for ob in ledger.children(exp["record_id"], "observation"):
        arts = ledger.children(ob["record_id"], "artifact")
        used += [ob["record_id"]] + [a["record_id"] for a in arts]
        lines.append(f"Stream {ob['body']['modality']}: {media.add(ob, arts)}.")
    return lines, media, used


def _failure_answer(ledger, exp):
    out = []
    for st in ledger.children(exp["record_id"], "step"):
        tags = set(st["body"].get("failure_tags", []))
        for ob in ledger.children(st["record_id"], "observation"):
            tags |= set(ob["body"].get("failure_tags", []))
        if st["body"]["status"] == "failed" or tags:
            out.append(f"Step {st['body']['index']}: " + ", ".join(sorted(tags)))
    return "; ".join(out) if out else "none"


def to_training(ledger, allow_synthetic=False, sealed_ids=(), seed=0):
    """Return a list of coscientist.v1 records, two per finished experiment."""
    problems = ledger.verify()
    if problems:
        raise LedgerError("ledger does not verify: " + "; ".join(problems[:5]))
    head = ledger.events[-1]["event_hash"] if ledger.events else None
    sealed = set(sealed_ids)
    out = []
    for exp in sorted(ledger.current().values(), key=lambda e: e["seq"]):
        if exp["level"] != "experiment" or exp["body"]["status"] not in DONE:
            continue
        if exp["record_id"] in sealed:
            continue
        synthetic = exp["origin"] == "synthetic_demo"
        if synthetic and not allow_synthetic:
            continue
        lines, media, used = _render(ledger, exp, synthetic)
        if not lines or sealed & set(used):
            continue
        b = exp["body"]
        record = (f"Experiment {exp['record_id']} ({b['title']}, protocol {b['protocol_id']}, "
                  f"status {b['status']}).\n" + "\n".join(lines))
        lic = exp["license"]
        base = {
            "schema": "coscientist.v1",
            "geoms": [], "images": [], "sims": [], "audios": [],
            "pair": None,
            "source": {"origin": exp["origin"], "license": lic, "commercial_ok": lic in COMMERCIAL_OK},
            "provenance": {"generator": "labledger.export", "verified_by": "exact", "twin_of": None},
            "ledger": {"experiment_id": exp["record_id"], "record_ids": used, "head_hash": head},
        }
        qa = dict(base, id=f"{exp['record_id']}:failures", task="protocol_qa",
                  geoms=media.geoms, images=media.images, sims=media.sims, audios=media.audios,
                  messages=[{"role": "user", "content": record + "\nWhich steps failed or were "
                             "flagged, and with which failure tags? Answer 'none' if none."},
                            {"role": "assistant", "content": _failure_answer(ledger, exp)}])
        qa["verifier"] = {"kind": "exact", "spec": {"answer": qa["messages"][1]["content"]}}
        out.append(qa)
        steps = ledger.children(exp["record_id"], "step")
        if len(steps) >= 3:
            rng = random.Random(int(hashlib.sha256(f"{seed}:{exp['record_id']}".encode())
                                    .hexdigest()[:8], 16))
            order = list(range(len(steps)))
            while order == sorted(order):
                rng.shuffle(order)
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            shown = "\n".join(f"{letters[j]}. {steps[i]['body']['action']}" for j, i in enumerate(order))
            answer = " ".join(letters[order.index(i)] for i in range(len(steps)))
            od = dict(base, id=f"{exp['record_id']}:order", task="order",
                      messages=[{"role": "user", "content": "Put these protocol steps in the "
                                 f"order they were run:\n{shown}"},
                                {"role": "assistant", "content": answer}],
                      verifier={"kind": "exact", "spec": {"answer": answer}})
            out.append(od)
    return out


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in records)


PLACEHOLDER_KINDS = {"geom": "geoms", "img": "images", "sim": "sims", "audio": "audios"}


def check_training_record(rec):
    """Every <kind:i> placeholder must match exactly the indices 0..len-1 of its array."""
    import re

    used = {k: set() for k in PLACEHOLDER_KINDS}
    for m in rec["messages"]:
        for kind, i in re.findall(r"<(geom|img|sim|audio):(\d+)>", m["content"]):
            used[kind].add(int(i))
    for kind, key in PLACEHOLDER_KINDS.items():
        if used[kind] != set(range(len(rec.get(key, [])))):
            raise ValueError(f"{rec['id']}: <{kind}:i> {sorted(used[kind])} vs {len(rec[key])} {key}")


def audio_text(a, i):
    """Text for <audio:i>, for collators that inline it (like <img:i> and <sim:i> in PR #86)."""
    flag = "flagged" if a["score"] > a["threshold"] else "normal"
    return (f"[audio {i}: {a['instrument_id']}, {a['mic']} mic, {a['duration_s']:g} s, "
            f"score={a['score']:.2f} vs threshold={a['threshold']:.2f}, {flag}]")

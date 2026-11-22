"""Protocol-QA eval stub: score predictions against a JSONL item file.

Item (one JSON object per line):
    {"id": "...", "task": "mcq" | "exact" | "order",
     "question": "...", "choices": [...]   # mcq only
     "answer": "B" | "text" | [step ids in order],
     "molecules": ["CCO", ...]             # optional SMILES context}

Prediction (one JSON object per line): {"id": "...", "pred": ...}

Scoring:
    mcq   -> correct if pred letter or pred text matches the answer choice.
    exact -> correct if normalized strings match.
    order -> correct if the full order matches (list or "s1, s2" string);
             also reports pairwise order accuracy.

Usage:
    python -m lab.protocol_qa --items lab/fixtures/protocol_qa_tiny.jsonl --preds preds.jsonl
    python -m lab.protocol_qa --items lab/fixtures/protocol_qa_tiny.jsonl --print-prompts

Real data: LAB-Bench ProtocolQA and BioProBench tasks are loaded by
github.com/yaniguan/labbench-posttrain (labeval/, bioenvs/). This file only scores.
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional, Sequence

TASKS = ("mcq", "exact", "order")
LETTERS = string.ascii_uppercase


def norm(s) -> str:
    """Lowercase, trim, drop surrounding quotes/period, collapse spaces."""
    s = str(s).strip().strip("'\"").rstrip(".").lower()
    return re.sub(r"\s+", " ", s)


def load_jsonl(path) -> list[dict]:
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: bad JSON ({e})") from e
    return rows


def validate_item(item: dict) -> None:
    for key in ("id", "task", "question", "answer"):
        if key not in item:
            raise ValueError(f"item missing '{key}': {item}")
    task = item["task"]
    if task not in TASKS:
        raise ValueError(f"item {item['id']}: unknown task '{task}'")
    if task == "mcq":
        choices = item.get("choices") or []
        if not (2 <= len(choices) <= len(LETTERS)):
            raise ValueError(f"item {item['id']}: mcq needs 2..{len(LETTERS)} choices")
        if mcq_index(item["answer"], choices) is None:
            raise ValueError(f"item {item['id']}: answer not in choices")
    if task == "order":
        ans = item["answer"]
        if not isinstance(ans, list) or len(ans) < 2 or len(set(ans)) != len(ans):
            raise ValueError(f"item {item['id']}: order answer must be a list of unique ids")


def mcq_index(value, choices: Sequence[str]) -> Optional[int]:
    """Map a letter ('B', '(B)', 'b)') or the choice text to a choice index."""
    if value is None:
        return None
    v = str(value).strip()
    by_text = {norm(c): i for i, c in enumerate(choices)}
    if any(len(norm(c)) == 1 for c in choices) and re.fullmatch(r"[A-Za-z]", v) and norm(v) in by_text:
        if [c.strip().upper() for c in choices] != list(LETTERS[: len(choices)]):
            return by_text[norm(v)]
    m = re.fullmatch(r"\(?([A-Za-z])\)?[.):]?", v)
    if m:
        idx = LETTERS.index(m.group(1).upper())
        if idx < len(choices):
            return idx
    return by_text.get(norm(v))


def pairwise_order_accuracy(pred: Sequence, gold: Sequence) -> float:
    """Share of gold step pairs that keep the same relative order in pred.

    Steps missing from pred count as wrong pairs.
    """
    pos = {s: i for i, s in enumerate(pred)}
    pairs = list(combinations(gold, 2))
    if not pairs:
        return 0.0
    ok = sum(1 for a, b in pairs if a in pos and b in pos and pos[a] < pos[b])
    return ok / len(pairs)


def score_item(item: dict, pred) -> dict:
    task = item["task"]
    out = {"id": item["id"], "task": task, "answered": pred is not None}
    if task == "mcq":
        gold = mcq_index(item["answer"], item["choices"])
        out["correct"] = pred is not None and mcq_index(pred, item["choices"]) == gold
    elif task == "exact":
        out["correct"] = pred is not None and norm(pred) == norm(item["answer"])
    else:  # order
        if isinstance(pred, str):  # "s1, s2, s3" as asked by build_prompt
            pred = [x.strip() for x in pred.split(",") if x.strip()]
        seq = list(pred) if isinstance(pred, (list, tuple)) else []
        out["correct"] = seq == list(item["answer"])
        out["pairwise"] = pairwise_order_accuracy(seq, item["answer"])
    return out


def score(items: Iterable[dict], preds: Iterable[dict]) -> dict:
    """Score predictions. Missing predictions count as unanswered and wrong."""
    items = list(items)
    for it in items:
        validate_item(it)
    by_id = {}
    for p in preds:
        if "id" not in p:
            raise ValueError(f"prediction missing 'id': {p}")
        by_id[p["id"]] = p.get("pred")
    rows = [score_item(it, by_id.get(it["id"])) for it in items]

    def summary(sub: list[dict]) -> dict:
        n = len(sub)
        s = {
            "n": n,
            "answered": sum(r["answered"] for r in sub),
            "correct": sum(r["correct"] for r in sub),
            "accuracy": (sum(r["correct"] for r in sub) / n) if n else None,
        }
        pw = [r["pairwise"] for r in sub if "pairwise" in r]
        if pw:
            s["pairwise_order_accuracy"] = sum(pw) / len(pw)
        return s

    by_task = {t: summary([r for r in rows if r["task"] == t]) for t in TASKS}
    return {
        "overall": summary(rows),
        "by_task": {t: v for t, v in by_task.items() if v["n"]},
        "unknown_pred_ids": sorted(set(by_id) - {it["id"] for it in items}),
        "rows": rows,
    }


def build_prompt(item: dict, molecule_block: str = "", physics_lines: Sequence[str] = ()) -> str:
    """Plain prompt text for a reasoner.

    molecule_block: from lab.molecule_context (OCSR, the "eyes").
    physics_lines: `[physics] ...` lines from WignerFlow `lab/surrogate_query.to_reasoner_context`.
    """
    parts = []
    if molecule_block:
        parts.append(molecule_block.rstrip())
    elif item.get("molecules"):
        parts.append("Molecules: " + ", ".join(str(m) for m in item["molecules"]))
    if physics_lines:
        parts.append("\n".join(physics_lines))
    parts.append(f"Question:\n{item['question']}")
    if item["task"] == "mcq":
        parts.append("Choices:\n" + "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(item["choices"])))
        parts.append("Answer with one letter.")
    elif item["task"] == "order":
        parts.append("Answer with the step ids in the correct order, comma-separated.")
    else:
        parts.append("Answer with a short exact value.")
    return "\n\n".join(parts) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items", required=True, help="JSONL item file")
    ap.add_argument("--preds", help="JSONL prediction file")
    ap.add_argument("--print-prompts", action="store_true", help="print prompts and exit")
    ap.add_argument("--out", help="write full score JSON here")
    args = ap.parse_args(argv)

    items = load_jsonl(args.items)
    if args.print_prompts:
        for it in items:
            validate_item(it)
            print(f"### {it['id']}\n{build_prompt(it)}")
        return 0
    if not args.preds:
        ap.error("--preds is required unless --print-prompts is set")
    res = score(items, load_jsonl(args.preds))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    json.dump({k: v for k, v in res.items() if k != "rows"}, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

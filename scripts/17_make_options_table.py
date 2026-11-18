"""Build a standalone multi-seed Option A/B/C table on the official TensorQA-v0 file.

Keeps the packaged-v0 32-rotation comparison separate from the older 4-rotation Colab table
(`docs/results_options_abc.md`) and verifies each run's `provenance` block.

Run:
  python scripts/17_make_options_table.py runs --out results_options_v0.md --json results_options_v0.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.provenance import (  # noqa: E402
    OFFICIAL_V0_SPLIT_COUNTS,
    OFFICIAL_V0_UNPACKED_JSON_SHA256,
)


ROW_ORDER = [
    "A",
    "B",
    "B_attn_sup",
    "B_normalized",
    "C",
    "C_channelwise",
]


def classify_option_run(args: dict) -> tuple[str, str, str] | None:
    """Return `(row_id, row_label, text_name)` for an Option A/B/C run, or `None` to skip."""
    if "llm" in args or "text" not in args:
        return None
    if args.get("model", "ylm") != "ylm":
        return None
    if args.get("text") == "taskid":
        return None
    if float(args.get("train_fraction", 1.0)) != 1.0:
        return None
    if args.get("legacy_odd_scalars", False):
        return None

    opt = str(args.get("option", "A")).upper()
    text = str(args["text"])
    attn_sup = float(args.get("attn_sup", 0.0) or 0.0)
    b_mode = str(args.get("b_score_mode", "legacy"))
    c_mode = str(args.get("c_tp_mode", "fully_connected"))

    if opt == "A" and attn_sup == 0.0:
        return ("A", "Option A", text)
    if opt == "B":
        if attn_sup > 0.0:
            return ("B_attn_sup", f"Option B (`--attn_sup {attn_sup:g}`)", text)
        if b_mode != "legacy":
            return ("B_normalized", f"Option B (`{b_mode}`)", text)
        return ("B", "Option B", text)
    if opt == "C":
        if c_mode != "fully_connected":
            return ("C_channelwise", f"Option C (`{c_mode}`)", text)
        return ("C", "Option C", text)
    return None


def verify_v0_option_provenance(result: dict, path: str, *, expected_rotations: int = 32) -> None:
    """Raise `ValueError` if `result` does not match the packaged v0 32-rotation protocol."""
    prov = result.get("provenance")
    if not isinstance(prov, dict):
        raise ValueError(f"{path}: missing `provenance` block in results.json")
    q_sha = prov.get("questions_sha256")
    if q_sha != OFFICIAL_V0_UNPACKED_JSON_SHA256:
        raise ValueError(
            f"{path}: `questions_sha256` mismatch (expected official v0 unpacked SHA-256 "
            f"{OFFICIAL_V0_UNPACKED_JSON_SHA256}, got {q_sha})"
        )
    if prov.get("split_counts") != OFFICIAL_V0_SPLIT_COUNTS:
        raise ValueError(
            f"{path}: `split_counts` mismatch (expected {OFFICIAL_V0_SPLIT_COUNTS}, got {prov.get('split_counts')})"
        )
    if int(prov.get("eval_rotations", 0)) != expected_rotations:
        raise ValueError(
            f"{path}: `eval_rotations` must be {expected_rotations}, got {prov.get('eval_rotations')}"
        )
    if bool(prov.get("eval_reflections", False)):
        raise ValueError(f"{path}: `eval_reflections` must be False for the v0 32-rotation protocol")


def _fmt(values: list[float], digits: int) -> str:
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values):.{digits}f}"


def build_options_table(
    folders: list[str],
    *,
    out_md: str | os.PathLike = "results_options_v0.md",
    out_json: str | os.PathLike | None = None,
    min_seeds: int = 3,
    expected_rotations: int = 32,
    allow_unverified: bool = False,
) -> tuple[str, dict]:
    grouped: dict[tuple[str, str, str], dict[int, dict]] = defaultdict(dict)
    for folder in folders:
        for path in sorted(glob.glob(f"{folder}/*/results.json")):
            with open(path, encoding="utf-8") as handle:
                res = json.load(handle)
            cls = classify_option_run(res.get("args", {}))
            if cls is None:
                continue
            if not allow_unverified:
                verify_v0_option_provenance(res, path, expected_rotations=expected_rotations)
            seed = int(res["args"].get("seed", 0))
            if seed in grouped[cls]:
                raise ValueError(f"Duplicate seed {seed} for {cls[1]} ({cls[2]}) at {path}")
            grouped[cls][seed] = res

    if not grouped:
        raise ValueError(f"No matching Option A/B/C runs found in {folders}")

    for (_, label, text), by_seed in grouped.items():
        if len(by_seed) < min_seeds:
            raise ValueError(
                f"{label} ({text}) has {len(by_seed)} seed(s); at least {min_seeds} seeds required"
            )

    sorted_keys = sorted(
        grouped.keys(),
        key=lambda k: (k[2], ROW_ORDER.index(k[0]) if k[0] in ROW_ORDER else 99, k[1]),
    )

    proto_note = (
        f"Official TensorQA-v0 (`84,757` questions, `584` unique strings; SHA-256 `{OFFICIAL_V0_UNPACKED_JSON_SHA256[:12]}…`), "
        f"`{expected_rotations}` random SO(3) rotations per test question, mean ± std over seeds."
        if not allow_unverified
        else "Unverified smoke/custom protocol (provenance check bypassed via `--allow-unverified`)."
    )
    lines = [
        "# Options A / B / C on Official TensorQA-v0",
        "",
        proto_note,
        "Kept separate from the Oct 30–31 single-seed 4-rotation + reflection Colab table in `docs/results_options_abc.md`.",
        "",
    ]

    summary_json: dict = {
        "protocol": {
            "official_v0_verified": not allow_unverified,
            "questions_sha256": OFFICIAL_V0_UNPACKED_JSON_SHA256 if not allow_unverified else None,
            "eval_rotations": expected_rotations,
            "eval_reflections": False,
            "min_seeds": min_seeds,
        },
        "rows": [],
    }

    for split, title in [
        ("test", "Test (wordings seen in training)"),
        ("test_heldout", "Test, held-out wordings"),
    ]:
        lines += [
            f"## {title}",
            "",
            "| Option | text | seeds | dipole MAE | dipole cos | force MAE | force cos | atom acc | eq_err |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for key in sorted_keys:
            row_id, label, text = key
            runs = [grouped[key][s] for s in sorted(grouped[key])]
            d = [r[split]["dipole"] for r in runs]
            f = [r[split]["force"] for r in runs]
            eq_max = max(
                max(x.get("eq_err_max") or 0.0, y.get("eq_err_max") or 0.0)
                for x, y in zip(d, f)
            )
            lines.append(
                "| "
                + " | ".join([
                    label,
                    text,
                    str(len(runs)),
                    _fmt([x["mae"] for x in d], 3),
                    _fmt([x["cosine"] for x in d], 3),
                    _fmt([x["mae"] for x in f], 4),
                    _fmt([x["cosine"] for x in f], 3),
                    _fmt([x["atom_acc"] for x in f], 3),
                    f"{eq_max:.0e}",
                ])
                + " |"
            )
        lines.append("")

    for key in sorted_keys:
        row_id, label, text = key
        seeds = sorted(grouped[key])
        runs = [grouped[key][s] for s in seeds]
        entry = {
            "row_id": row_id,
            "label": label,
            "text": text,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "questions_sha256": sorted({
                (r.get("provenance") or {}).get("questions_sha256") or ""
                for r in runs
            }),
        }
        for split in ("test", "test_heldout"):
            entry[split] = {}
            for task in ("dipole", "force"):
                metrics = ("mae", "cosine", "atom_acc") if task == "force" else ("mae", "cosine")
                entry[split][task] = {
                    m: {
                        "mean": float(np.mean([r[split][task][m] for r in runs])),
                        "std": float(np.std([r[split][task][m] for r in runs])),
                    }
                    for m in metrics
                }
        summary_json["rows"].append(entry)

    md_text = "\n".join(lines)
    Path(out_md).write_text(md_text + "\n", encoding="utf-8")
    if out_json is not None:
        Path(out_json).write_text(json.dumps(summary_json, indent=2) + "\n", encoding="utf-8")
    return md_text, summary_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build standalone multi-seed Option A/B/C table on official TensorQA-v0.")
    parser.add_argument("folders", nargs="*", default=["runs"], help="Run folders containing */results.json")
    parser.add_argument("--out", default="results_options_v0.md", help="Output Markdown path")
    parser.add_argument("--json", default=None, help="Optional JSON summary output path")
    parser.add_argument("--min-seeds", type=int, default=3, help="Minimum seed count required per row")
    parser.add_argument("--eval-rotations", type=int, default=32, help="Required eval_rotations in provenance")
    parser.add_argument("--allow-unverified", action="store_true", help="Skip official-v0 provenance hash checks (smoke runs only)")
    args = parser.parse_args(argv)
    md_text, _ = build_options_table(
        args.folders or ["runs"],
        out_md=args.out,
        out_json=args.json,
        min_seeds=args.min_seeds,
        expected_rotations=args.eval_rotations,
        allow_unverified=args.allow_unverified,
    )
    print(md_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

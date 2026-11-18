"""Collect runs/*/results.json into one table (mean +- std over seeds).

Output: results.md (and printed)

Run:
  python scripts/07_make_table.py                        # all runs in runs/
  python scripts/07_make_table.py runs_slurm runs        # combine folders
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json

import numpy as np

NAMES = {
    "minilm": "YLM (Option A) + MiniLM",
    "matscibert": "YLM (Option A) + MatSciBERT",
    "taskid": "Baseline v: task ID instead of text",
    "llm_noaug": "Baseline ii: coordinates as text, Qwen2.5-0.5B",
    "llm_aug": "Baseline ii: same + rotation augmentation",
}


def group_key(args: dict):
    if "llm" in args:
        return ("llm_aug" if args.get("augment") else "llm_noaug"), 1.0
    if args.get("model", "ylm") != "ylm":
        return None
    if args.get("option", "A") != "A":
        return None
    if float(args.get("attn_sup", 0.0) or 0.0) > 0.0:
        return None
    if args.get("legacy_odd_scalars", False):
        return None
    if int(args.get("n_cross_layers", 1) or 1) != 1:
        return None
    return args["text"], float(args.get("train_fraction", 1.0))


def fmt(values, digits):
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values):.{digits}f}"


def fmt_sci(values):
    values = [v for v in values if v is not None]
    return f"{np.max(values):.0e}" if values else "-"


def build_table(folders: list[str], out_path: str = "results.md") -> str:
    groups = defaultdict(list)
    for folder in folders:
        for path in sorted(glob.glob(f"{folder}/*/results.json")):
            with open(path, encoding="utf-8") as handle:
                r = json.load(handle)
            key = group_key(r["args"])
            if key is not None:
                groups[key].append(r)

    lines = [
        "# Results: TensorQA-v0",
        "",
        "Mean ± std over seeds. 32 random rotations per test question (8 for the LLM baseline).",
        "MAE units: dipole e·bohr, force hartree/bohr. eq_err = max over test questions of ||f(Rx) - R f(x)|| / ||f(x)||.",
        "",
    ]
    for split in ["test", "test_heldout"]:
        title = "Test (wordings seen in training)" if split == "test" else "Test, held-out wordings"
        lines += [
            f"## {title}",
            "",
            "| Model | data | seeds | dipole MAE | dipole cos | force MAE | force cos | atom acc | dipole MAE, orig. frame | eq_err |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        order = ["minilm", "matscibert", "taskid", "llm_noaug", "llm_aug"]
        for key in sorted(groups, key=lambda k: (order.index(k[0]) if k[0] in order else 99, -k[1])):
            rs = [r[split] for r in groups[key] if split in r]
            if not rs:
                continue
            d = [r["dipole"] for r in rs]
            f = [r["force"] for r in rs]
            lines.append(
                "| "
                + " | ".join([
                    NAMES.get(key[0], key[0]),
                    f"{key[1]:.0%}",
                    str(len(rs)),
                    fmt([x["mae"] for x in d], 3),
                    fmt([x["cosine"] for x in d], 3),
                    fmt([x["mae"] for x in f], 4),
                    fmt([x["cosine"] for x in f], 3),
                    fmt([x.get("atom_acc") for x in f], 3),
                    fmt([x.get("mae_orig_frame") for x in d], 3),
                    fmt_sci([max(x["eq_err_max"] or 0, y["eq_err_max"] or 0) for x, y in zip(d, f)]),
                ])
                + " |"
            )
        lines.append("")
    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    return text


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Build TensorQA-v0 results.md table.")
    parser.add_argument("folders", nargs="*", default=["runs"], help="Run folders containing */results.json")
    parser.add_argument("--out", default="results.md", help="Markdown file to write")
    args = parser.parse_args(argv)
    text = build_table(args.folders or ["runs"], out_path=args.out)
    print(text)


if __name__ == "__main__":
    main()

"""Collect runs/*/results.json into one table (mean +- std over seeds).

Output: results.md (and printed)

Run:
  python scripts/07_make_table.py            # all runs in runs/
  python scripts/07_make_table.py runs_expanse runs_mac   # several folders
"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

NAMES = {
    "minilm": "YLM (Option A) + MiniLM",
    "matscibert": "YLM (Option A) + MatSciBERT",
    "taskid": "Baseline v: task ID instead of text",
    "llm_noaug": "Baseline ii: coordinates as text, Qwen2.5-0.5B",
    "llm_aug": "Baseline ii: same + rotation augmentation",
}


def group_key(args):
    if "llm" in args:
        return ("llm_aug" if args.get("augment") else "llm_noaug"), 1.0
    return args["text"], args.get("train_fraction", 1.0)


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


def main():
    folders = sys.argv[1:] or ["runs"]
    groups = defaultdict(list)
    for folder in folders:
        for path in sorted(glob.glob(f"{folder}/*/results.json")):
            r = json.load(open(path))
            groups[group_key(r["args"])].append(r)

    lines = ["# Results: TensorQA-v0", "",
             "Mean ± std over seeds. 32 random rotations per test question (8 for the LLM baseline).",
             "MAE units: dipole e·bohr, force hartree/bohr. eq_err = max over test questions of ||f(Rx) - R f(x)|| / ||f(x)||.",
             ""]
    for split in ["test", "test_heldout"]:
        title = "Test (wordings seen in training)" if split == "test" else "Test, held-out wordings"
        lines += [f"## {title}", "",
                  "| Model | data | seeds | dipole MAE | dipole cos | force MAE | force cos | atom acc | dipole MAE, orig. frame | eq_err |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        order = ["minilm", "matscibert", "taskid", "llm_noaug", "llm_aug"]
        for key in sorted(groups, key=lambda k: (order.index(k[0]) if k[0] in order else 99, -k[1])):
            rs = [r[split] for r in groups[key] if split in r]
            if not rs:
                continue
            d = [r["dipole"] for r in rs]
            f = [r["force"] for r in rs]
            lines.append("| " + " | ".join([
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
            ]) + " |")
        lines.append("")
    text = "\n".join(lines)
    open("results.md", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

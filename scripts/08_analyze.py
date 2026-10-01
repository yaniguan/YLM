"""Analysis for v0: where does the model succeed or fail?

Uses the saved models runs/<name>/model.pt (YLM and task-ID runs) and runs/*/results.json.

Output (all in analysis/):
  sample_efficiency.png   force direction (cosine) and force MAE vs fraction of training data
  breakdown.md            force questions split by element, molecule size, held-out wording;
                          the atom descriptions that fail most often

Run (after copying runs back from the cluster):
  python scripts/08_analyze.py --runs runs
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.model import YLM  # noqa: E402

# reuse the data loading from 04_train.py (file name starts with a digit, so import it by path)
_spec = importlib.util.spec_from_file_location("train", os.path.join(ROOT, "scripts", "04_train.py"))
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

ELEMENT = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
LABEL = {"minilm": "YLM + MiniLM", "matscibert": "YLM + MatSciBERT", "taskid": "task ID (no text)"}


def sample_efficiency(runs_dir, out_dir):
    points = defaultdict(lambda: defaultdict(list))   # text -> fraction -> [(force_cosine, force_mae)]
    for path in glob.glob(f"{runs_dir}/*/results.json"):
        r = json.load(open(path))
        a = r["args"]
        if "text" not in a:
            continue  # LLM baseline
        f = r["test"]["force"]
        points[a["text"]][a.get("train_fraction", 1.0)].append((f["cosine"], f["mae"]))
    if not points:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for text, by_frac in sorted(points.items()):
        fr = sorted(by_frac)
        acc = [np.mean([p[0] for p in by_frac[x]]) for x in fr]
        mae = [np.mean([p[1] for p in by_frac[x]]) for x in fr]
        axes[0].plot(fr, acc, marker="o", label=LABEL.get(text, text))
        axes[1].plot(fr, mae, marker="o", label=LABEL.get(text, text))
    for ax, name in zip(axes, ["force direction (cosine), test", "force MAE (hartree/bohr), test"]):
        ax.set_xscale("log")
        ax.set_xlabel("fraction of training questions")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("100%: mean of 4-5 seeds. 3% / 10% / 30%: one seed each.", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/sample_efficiency.png", dpi=150)
    print("wrote", f"{out_dir}/sample_efficiency.png")


@torch.no_grad()
def predict(model, data, split, scale, batch_size=512):
    """Predictions in the original frame (equivariance is checked separately in results.json)."""
    """Also returns, per question:
      hit       attention peak is on the asked atom
      hit_near  attention peak is on the asked atom OR an atom bonded to it (closer than 3.4 bohr = 1.8 A)
      mass      share of attention on the asked atom"""
    ys, hits, nears, masses = [], [], [], []
    for start in range(0, len(split["mol"]), batch_size):
        idx = torch.arange(start, min(start + batch_size, len(split["mol"])))
        inputs, target = train.get_batch(data, split, idx, "cpu")
        y, attn = model(*inputs)
        ys.append(y * scale[inputs[5]][:, None])
        atom = split["atom"][idx].clamp(min=0)                     # dipole questions have atom = -1
        peak = attn.argmax(-1)
        pos = inputs[1]
        b = torch.arange(len(idx))
        d_peak = (pos[b, peak] - pos[b, atom]).norm(dim=-1)
        hits.append(peak == split["atom"][idx])
        nears.append(d_peak < 3.4)
        masses.append(attn[b, atom])
    return torch.cat(ys), torch.cat(hits), torch.cat(nears), torch.cat(masses)


def breakdown(runs_dir, out_dir):
    questions = json.load(open("data/tensorqa_v0.json"))
    labels = None
    lines = ["# Breakdown of force questions (seed 0, full data)", ""]
    for text in ["minilm", "matscibert", "taskid"]:
        run = f"{runs_dir}/m1_{text}_s0"
        if not os.path.exists(f"{run}/results.json"):  # still training or never ran
            continue
        text_path = None if text == "taskid" else f"data/text_{text}.pt"
        data, splits = train.load_data("data/qm9_xtb.pkl", "data/tensorqa_v0.json", text_path)
        if labels is None:
            import pickle
            labels = pickle.load(open("data/qm9_xtb.pkl", "rb"))
        scale = torch.tensor(json.load(open(f"{run}/results.json"))["scale"])
        model = YLM(text_dim=data["text_dim"], text_mode="taskid" if text == "taskid" else "text")
        model.load_state_dict(torch.load(f"{run}/model.pt", map_location="cpu"))
        model.eval()

        lines += [f"## {LABEL[text]}", ""]
        for split_name in ["test", "test_heldout"]:
            split = splits[split_name]
            y, hit, near, mass = predict(model, data, split, scale)
            qs = [q for q in questions if q["split"] == split_name]
            is_force = split["task"] == 1
            err = (y - split["target"]).abs().mean(-1)

            groups = defaultdict(list)
            for k, q in enumerate(qs):
                if q["task"] != "force":
                    continue
                lab = labels[q["mol"]]
                z = int(lab["numbers"][q["atom"]])
                n = len(lab["numbers"])
                size = "<=12 atoms" if n <= 12 else ("13-18 atoms" if n <= 18 else ">18 atoms")
                ring = "ring words" if "ring" in q["question"] else "no ring words"
                for key in [f"element {ELEMENT[z]}", size, ring]:
                    groups[key].append((hit[k].item(), near[k].item(), mass[k].item(), err[k].item()))
            lines += [f"### {split_name}: {int(is_force.sum())} force questions", "",
                      f"- attention peak on the asked atom: {hit[is_force].float().mean():.3f}",
                      f"- peak on the asked atom or a bonded neighbour: {near[is_force].float().mean():.3f}",
                      f"- mean share of attention on the asked atom: {mass[is_force].mean():.3f}",
                      f"- force MAE: {err[is_force].mean():.4f}", "",
                      "| group | n | peak on atom | peak on atom or neighbour | attention share | force MAE |",
                      "|---|---|---|---|---|---|"]
            for key in sorted(groups):
                g = np.array(groups[key])
                lines.append(f"| {key} | {len(g)} | {g[:, 0].mean():.3f} | {g[:, 1].mean():.3f} | "
                             f"{g[:, 2].mean():.3f} | {g[:, 3].mean():.4f} |")
            lines.append("")

            if split_name == "test" and text != "taskid":
                # which atom descriptions fail most (at least 20 questions)
                desc = defaultdict(list)
                for k, q in enumerate(qs):
                    if q["task"] == "force":
                        # "...the oxygen atom bonded to two carbons being pushed?" -> "oxygen atom bonded to two carbons"
                        d = re.search(r"the (\w+ atom[^?.,]*)", q["question"]).group(1)
                        d = re.sub(r" (being pushed|pulled)$", "", d)
                        desc[d].append(hit[k].item())
                worst = sorted([(np.mean(v), len(v), d) for d, v in desc.items() if len(v) >= 20])[:10]
                lines += ["Atom descriptions with the lowest attention accuracy (>= 20 questions):", "",
                          "| description | n | atom acc |", "|---|---|---|"]
                lines += [f"| {d} | {n} | {a:.3f} |" for a, n, d in worst]
                lines.append("")
    open(f"{out_dir}/breakdown.md", "w").write("\n".join(lines) + "\n")
    print("wrote", f"{out_dir}/breakdown.md")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="analysis")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sample_efficiency(args.runs, args.out)
    breakdown(args.runs, args.out)


if __name__ == "__main__":
    main()

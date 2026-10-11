"""Evaluate a packaged v0 checkpoint on TensorQA-v0 and show a rotation demo.

  python scripts/09_eval_checkpoint.py --ckpt checkpoints/v0/ylm_matscibert_s0.pt
  python scripts/09_eval_checkpoint.py --ckpt checkpoints/v0/ylm_matscibert_s0.pt --demo

Needs data/qm9_xtb.pkl, data/tensorqa_v0.json and data/text_<name>.pt (steps 1-3 of "How to run").
Checkpoints are made by scripts/10_package_checkpoints.py.
"""
import argparse
import importlib.util
import json
import os
import sys

import torch
from e3nn import o3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.model import YLM  # noqa: E402

spec = importlib.util.spec_from_file_location("train04", os.path.join(ROOT, "scripts", "04_train.py"))
train04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train04)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--splits", nargs="+", default=["test", "test_heldout"])
    p.add_argument("--eval_rotations", type=int, default=32)
    p.add_argument("--demo", action="store_true", help="print one force question under a random rotation")
    p.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu")
    text = ck["text"]
    text_path = None if text == "taskid" else f"{args.data_dir}/text_{text}.pt"
    data, splits = train04.load_data(f"{args.data_dir}/qm9_xtb.pkl", f"{args.data_dir}/tensorqa_v0.json", text_path)
    model = YLM(text_dim=data["text_dim"], text_mode="taskid" if text == "taskid" else "text").to(device)
    model.load_state_dict(ck["state_dict"])
    scale = torch.tensor(ck["scale"])
    print(f"checkpoint {args.ckpt} | text={text} | seed={ck.get('seed')} | device={device}")

    for split in args.splits:
        r = train04.evaluate(model, data, splits[split], scale, device, n_rot=args.eval_rotations)
        print(split, json.dumps(r))

    if args.demo:
        questions = json.load(open(f"{args.data_dir}/tensorqa_v0.json"))
        q = next(q for q in questions if q["split"] == "test" and q["task"] == "force")
        print("\nQuestion:", q["question"])
        sp = splits["test"]
        idx = torch.tensor([next(i for i in range(len(sp["mol"])) if sp["atom"][i] == q["atom"] and sp["mol"][i] == q["mol"])])
        R = o3.rand_matrix()
        model.eval()
        with torch.no_grad():
            (y0, _), (y1, _) = [model(*train04.get_batch(data, sp, idx, device, Rm)[0]) for Rm in (torch.eye(3), R)]
        y0, y1 = (y0 * scale[1]).cpu()[0], (y1 * scale[1]).cpu()[0]
        print("force, original frame :", y0.tolist())
        print("force, molecule rotated:", y1.tolist())
        print("R @ original           :", (R @ y0).tolist())
        print("difference             :", (y1 - R @ y0).norm().item())


if __name__ == "__main__":
    main()

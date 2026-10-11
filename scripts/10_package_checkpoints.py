"""Bundle trained runs into small release checkpoints (state_dict + scale + metadata).

  python scripts/10_package_checkpoints.py --runs runs_expanse --out checkpoints/v0
"""
import argparse
import json
import os

import torch

NAMES = {  # packaged name -> run directory
    "ylm_matscibert_s0": "m1_matscibert_s0",
    "ylm_minilm_s0": "m1_minilm_s0",
    "baseline_taskid_s0": "m1_taskid_s0",
}

p = argparse.ArgumentParser()
p.add_argument("--runs", default="runs_expanse")
p.add_argument("--out", default="checkpoints/v0")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
summary = {}
for name, run in NAMES.items():
    res = json.load(open(f"{args.runs}/{run}/results.json"))
    sd = torch.load(f"{args.runs}/{run}/model.pt", map_location="cpu")
    torch.save({"state_dict": sd, "scale": res["scale"], "text": res["args"]["text"],
                "seed": res["args"]["seed"], "args": res["args"]}, f"{args.out}/{name}.pt")
    summary[name] = {"test": res["test"], "test_heldout": res["test_heldout"]}
    print("wrote", f"{args.out}/{name}.pt", os.path.getsize(f"{args.out}/{name}.pt") // 1024, "KB")
json.dump(summary, open(f"{args.out}/results.json", "w"), indent=2)

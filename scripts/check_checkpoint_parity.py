#!/usr/bin/env python3
"""Synthetic geometry O(3) diagnostic of released weights; NOT accuracy evaluation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src
import torch
from e3nn import o3
from src.model import YLM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('docs/checkpoint_parity_oct29b.json'))
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(29)
    rows = []
    for path in sorted(Path('checkpoints/v0').glob('*.pt')):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        d = ck['state_dict']['text_proj.0.weight'].numel()
        inputs = (torch.tensor([[6, 6, 8, 7, 1, 1, 1, 1]]), torch.randn(1, 8, 3)*2,
                  torch.ones(1, 8, dtype=torch.bool), torch.randn(1, 4, d),
                  torch.ones(1, 4, dtype=torch.bool), torch.tensor([1]))
        matrices = [('rotation', o3.rand_matrix()), ('reflection', -o3.rand_matrix())]
        for legacy in [True, False]:
            model = YLM(d, text_mode='taskid' if ck['text']=='taskid' else 'text',
                        legacy_odd_scalars=legacy).eval()
            model.load_state_dict(ck['state_dict'], strict=True)
            with torch.no_grad():
                y, a = model(*inputs)
                for name, R in matrices:
                    changed = list(inputs)
                    changed[1] = inputs[1] @ R.T
                    yr, ar = model(*changed)
                    delta = yr - y @ R.T
                    rows.append(dict(checkpoint=path.name, legacy_odd_scalars=legacy,
                        transform=name, eq_err_relative=(delta.norm()/y.norm().clamp_min(1e-12)).item(),
                        eq_err_max_abs=delta.abs().max().item(), attention_max_abs=(ar-a).abs().max().item()))
    result = dict(note='Synthetic geometry and random text states. No labels: no MAE/cosine accuracy claim.',
                  seed=29, threads=2, results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

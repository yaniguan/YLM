#!/usr/bin/env python3
"""Small synthetic CPU benchmark. Run from any directory; no data downloads."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src  # initialize older e3nn compatibility before importing o3
import torch
from e3nn import o3
from src.irreps_to_text import IrrepsToText


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atoms', type=int, nargs='+', default=[8, 32, 64])
    parser.add_argument('--tokens', type=int, nargs='+', default=[4, 16, 32])
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.atoms + args.tokens + [args.iterations, args.threads]) < 1 or args.warmup < 0:
        parser.error('sizes, iterations and threads must be positive; warmup must be nonnegative')
    torch.set_num_threads(args.threads)
    torch.manual_seed(29)
    irreps = o3.Irreps('16x0e + 4x0o + 8x1o + 4x1e + 4x2e')
    layer = IrrepsToText(32, irreps, irreps, key_dim=16).float()
    rows = []
    for n in args.atoms:
        for t in args.tokens:
            x = torch.randn(1, n, irreps.dim, requires_grad=True)
            h = torch.randn(1, t, 32, requires_grad=True)
            mask = torch.ones(1, n, dtype=torch.bool)
            def forward():
                with torch.no_grad():
                    layer(h, x, mask)
            def backward():
                layer.zero_grad(set_to_none=True)
                x.grad = h.grad = None
                layer(h, x, mask)[0].square().mean().backward()
            def measure(fn):
                for _ in range(args.warmup):
                    fn()
                start = time.perf_counter()
                for _ in range(args.iterations):
                    fn()
                return (time.perf_counter() - start) * 1000 / args.iterations
            f_ms, fb_ms = measure(forward), measure(backward)
            # Build each graph outside the timed interval for backward-only latency.
            b_elapsed = 0.0
            for step in range(args.warmup + args.iterations):
                layer.zero_grad(set_to_none=True)
                x.grad = h.grad = None
                loss = layer(h, x, mask)[0].square().mean()
                start = time.perf_counter()
                loss.backward()
                elapsed = time.perf_counter() - start
                if step >= args.warmup:
                    b_elapsed += elapsed
            b_ms = b_elapsed * 1000 / args.iterations
            with torch.no_grad():
                R = o3.rand_matrix()
                D = irreps.D_from_matrix(R)
                y = layer(h, x, mask)[0]
                yr = layer(h, x @ D.T, mask)[0]
                delta = yr - y @ D.T
                row = dict(atoms=n, tokens=t, forward_ms=f_ms,
                           backward_ms=b_ms, forward_backward_ms=fb_ms,
                           equivariance_max_abs=delta.abs().max().item(),
                           equivariance_relative_l2=(delta.norm()/y.norm().clamp_min(1e-12)).item())
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = dict(torch=torch.__version__, e3nn=__import__('e3nn').__version__,
                  threads=args.threads, dtype='float32', batch_size=1,
                  irreps=str(irreps), text_dim=32, key_dim=16,
                  iterations=args.iterations, warmup=args.warmup, results=rows)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

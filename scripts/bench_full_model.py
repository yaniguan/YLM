#!/usr/bin/env python3
"""CPU encoder + attention + head benchmark; each case has an isolated RSS peak.

Defaults use tiny widths to fit shared CPUs. --preset default uses YLM defaults.
Timings exclude initialization; RSS includes Python/PyTorch, initialization and graphs.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src  # allow older e3nn bundled constants under modern PyTorch
import torch
from e3nn import o3
from src.model import YLM

TINY = dict(hidden_irreps='8x0e + 2x0o + 4x1o + 2x1e + 2x2e',
            cross_irreps='4x0e + 4x1o + 2x1e + 2x2e', n_layers=3)


def case(args, option, n, t):
    torch.set_num_threads(args.threads)
    torch.manual_seed(29)
    model = YLM(32, option=option, **(TINY if args.preset == 'tiny' else {})).float()
    inputs = (torch.randint(1, 10, (1, n)), torch.randn(1, n, 3) * 2,
              torch.ones(1, n, dtype=torch.bool), torch.randn(1, t, 32),
              torch.ones(1, t, dtype=torch.bool), torch.zeros(1, dtype=torch.long))
    def forward():
        with torch.no_grad():
            model(*inputs)
    def both():
        model.zero_grad(set_to_none=True)
        model(*inputs)[0].square().mean().backward()
    def measure(fn):
        for _ in range(args.warmup):
            fn()
        start = time.perf_counter()
        for _ in range(args.iterations):
            fn()
        return 1000 * (time.perf_counter() - start) / args.iterations
    f, fb = measure(forward), measure(both)
    elapsed = 0
    for i in range(args.warmup + args.iterations):
        model.zero_grad(set_to_none=True)
        loss = model(*inputs)[0].square().mean()
        start = time.perf_counter()
        loss.backward()
        if i >= args.warmup:
            elapsed += time.perf_counter() - start
    errors = {}
    with torch.no_grad():
        y = model(*inputs)[0]
        for name, R in [('rotation', o3.rand_matrix()), ('reflection', -o3.rand_matrix())]:
            changed = list(inputs)
            changed[1] = inputs[1] @ R.T
            delta = model(*changed)[0] - y @ R.T
            errors[name] = dict(max_abs=delta.abs().max().item(),
                                relative_l2=(delta.norm()/y.norm().clamp_min(1e-12)).item())
    return dict(option=option, atoms=n, tokens=t, parameters=sum(p.numel() for p in model.parameters()),
                forward_ms=f, backward_ms=elapsed*1000/args.iterations, forward_backward_ms=fb,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                equivariance=errors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--atoms', nargs='+', type=int, default=[8, 24])
    p.add_argument('--tokens', nargs='+', type=int, default=[4, 16])
    p.add_argument('--options', nargs='+', choices=list('ABC'), default=list('ABC'))
    p.add_argument('--preset', choices=['tiny', 'default'], default='tiny')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--output', type=Path)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if min(args.atoms + args.tokens + [args.threads, args.iterations]) < 1 or args.warmup < 0:
        p.error('sizes/threads/iterations must be positive; warmup must be nonnegative')
    if args.worker:
        print(json.dumps(case(args, args.options[0], args.atoms[0], args.tokens[0])))
        return
    rows = []
    for option in args.options:
        for n in args.atoms:
            for t in args.tokens:
                command = [sys.executable, __file__, '--worker', '--options', option,
                           '--atoms', str(n), '--tokens', str(t), '--preset', args.preset,
                           '--threads', str(args.threads), '--warmup', str(args.warmup),
                           '--iterations', str(args.iterations)]
                env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
                row = json.loads(subprocess.check_output(command, env=env, text=True))
                rows.append(row)
                print(json.dumps(row), flush=True)
    result = dict(torch=torch.__version__, e3nn=__import__('e3nn').__version__,
                  preset=args.preset, batch_size=1, dtype='float32', threads=args.threads,
                  iterations=args.iterations, warmup=args.warmup,
                  memory='isolated process peak RSS (MiB), including runtime and initialization', results=rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

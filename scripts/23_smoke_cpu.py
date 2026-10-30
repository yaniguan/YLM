"""Bounded CPU integration driver. Writes actual outcomes; never fabricates smoke results."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('smiles', 'smiles_xyz', 'prefixA', 'xattnA', 'xattnB', 'xattnC')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='results/oct30_smoke')
    p.add_argument('--qm9_limit', type=int, default=2000)
    p.add_argument('--downstream_limit', type=int, default=400)
    p.add_argument('--tokens_budget', type=int, default=12000)
    p.add_argument('--sft_tokens', type=int, default=4000)
    p.add_argument('--max_minutes', type=float, default=38)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               TOKENIZERS_PARALLELISM='false')
    start = time.monotonic()
    stages = []
    runs = [str(out / 'runs' / arm) for arm in ARMS]

    def run_stage(name, command):
        remaining = args.max_minutes * 60 - (time.monotonic() - start)
        record = {'name': name, 'command': command}
        stages.append(record)
        if remaining <= 0:
            record.update(status='timeout', reason='Overall smoke deadline exhausted')
            raise TimeoutError(record['reason'])
        print('Running', name, flush=True)
        then = time.monotonic()
        try:
            with open(out / f'{name}.log', 'a') as log:
                result = subprocess.run([sys.executable] + command, cwd=ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
            record.update(status='ok' if result.returncode == 0 else 'failed', returncode=result.returncode)
            if result.returncode:
                raise RuntimeError(f'{name} failed; see {out / (name + ".log")}')
        except subprocess.TimeoutExpired:
            record.update(status='timeout')
            raise TimeoutError(f'{name} exceeded remaining smoke deadline') from None
        finally:
            record['seconds'] = time.monotonic() - then
            (out / 'stages.json').write_text(json.dumps(stages, indent=2))

    failure = None
    try:
        for arm, run in zip(ARMS, runs):
            command = ['scripts/20_pretrain_lm.py', '--arm', arm, '--out', run, '--device', 'cpu',
                       '--model', 'tiny', '--n_layer', '2', '--n_embd', '64', '--n_head', '4',
                       '--limit', str(args.qm9_limit), '--tokens_budget', str(args.tokens_budget),
                       '--max_len', '192', '--batch_size', '4', '--numeric_eval_n', '2', '--ckpt_every', '20']
            if args.resume and (Path(run) / 'config.json').exists():
                command.append('--resume')
            run_stage('pretrain_' + arm, command)
        for arm in ('smiles', 'xattnA'):
            dest = out / 'sft' / arm
            command = ['scripts/21_posttrain.py', '--pretrained', str(out / 'runs' / arm), '--out', str(dest),
                       '--device', 'cpu', '--tokens_budget', str(args.sft_tokens), '--max_len', '192',
                       '--batch_size', '4', '--numeric_eval_n', '2', '--ckpt_every', '20']
            if args.resume and (dest / 'config.json').exists():
                command.append('--resume')
            run_stage('sft_' + arm, command)
        for task in ('calisol', 'reddb'):
            run_stage('downstream_' + task, ['scripts/22_downstream_eval.py', '--runs'] + runs +
                      ['--tasks', task, '--out', str(out / task), '--device', 'cpu', '--probe_only',
                       '--limit', str(args.downstream_limit), '--max_len', '192', '--batch_size', '4',
                       '--seeds', '0', '1', '2', '--bootstrap', '200'])
    except (RuntimeError, TimeoutError) as error:
        failure = str(error)
    finally:
        summary = {'status': 'failed' if failure else 'ok', 'error': failure,
                   'wall_seconds': time.monotonic() - start, 'stages': stages,
                   'note': 'Integration smoke only; budgets/subsets do not establish scientific performance.'}
        (out / 'smoke.json').write_text(json.dumps(summary, indent=2))
        lines = ['# Oct-30 CPU smoke', '', f"Status: {summary['status']}", '',
                 f"Elapsed: {summary['wall_seconds'] / 60:.2f} minutes", '',
                 '| Stage | Status | Seconds |', '|---|---|---|']
        lines += [f"| {r['name']} | {r.get('status', 'failed')} | {r.get('seconds', 0):.1f} |" for r in stages]
        for arm in ARMS:
            path = out / 'runs' / arm / 'metrics.json'
            if path.exists():
                m = json.loads(path.read_text())
                lines += ['', f"{arm}: {m['tokens_seen']} tokens; validation loss={m['validation']['loss']}; "
                          f"parameters={m['params']['total']}."]
        for task in ('calisol', 'reddb'):
            path = out / task / 'SUMMARY.md'
            if path.exists():
                lines += ['', path.read_text()]
        lines += ['', failure or 'All requested integration stages completed.', '', summary['note'],
                  'A100 measurements: TBD (A100).']
        (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    if failure:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

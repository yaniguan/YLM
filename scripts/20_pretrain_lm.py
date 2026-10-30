"""Matched-token QM9 LM ablation. Run from any working directory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.profiling import Profiler
from src.ylm_lm import YLMConfig, YLMCausalLM
from src.tasks.common import (ARMS, seed_everything, pick_device, train_lm, save_run_model,
                              load_encoder_weights, make_logger)
from src.tasks.tokenizer import get_tokenizer
from src.tasks.qm9_pretrain import load_qm9, corpus
from src.tasks.evaluation import evaluate_lm, numeric_eval


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--tokens_budget', type=int, default=1000000)
    p.add_argument('--model', default='tiny', help='tiny, small, or hf:<name>')
    p.add_argument('--encoder', default='tiny', choices=['tiny', 'repo_tiny', 'small'])
    p.add_argument('--init_encoder')
    p.add_argument('--device')
    p.add_argument('--out', required=True)
    p.add_argument('--resume', action='store_true', help='Resume saved state if present; otherwise start fresh')
    p.add_argument('--profile_dir')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit', type=int)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--ckpt_every', type=int, default=100)
    p.add_argument('--stop_after_steps', type=int)
    p.add_argument('--numeric_eval_n', type=int, default=32)
    p.add_argument('--n_layer', type=int)
    p.add_argument('--n_embd', type=int)
    p.add_argument('--n_head', type=int)
    p.add_argument('--sdf', default='data/gdb9.sdf')
    p.add_argument('--qm9', default='data/qm9_xtb.pkl')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = make_logger(str(out / 'train.log'))
    seed_everything(args.seed)
    device = pick_device(args.device)
    tok = get_tokenizer(args.model)
    mode, xyz = ARMS[args.arm]
    rows = load_qm9(args.qm9, args.sdf, args.limit)
    splits = corpus(rows, args.seed, xyz)
    # Account for the largest prefix in context capacity; no arm-specific molecule filtering.
    max_atoms = max((len(r['geom']['Z']) for r in rows), default=0)
    cfg = YLMConfig(mode=mode, lm=args.model, vocab_size=tok.vocab_size,
                    bos_token_id=tok.bos_id, eos_token_id=tok.eos_id,
                    n_positions=max(1024, args.max_len + max_atoms), encoder=args.encoder,
                    n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head)
    config = {'model_cfg': cfg.to_dict(), 'args': vars(args),
              'splits': {k: [e.meta['qm9_index'] for e in v] for k, v in splits.items()}}
    if args.resume and (out / 'config.json').exists():
        previous = json.loads((out / 'config.json').read_text())
        for key in ('model_cfg', 'splits'):
            if previous[key] != config[key]:
                raise ValueError(f'Resume {key} differs from saved run')
        for key in ('tokens_budget', 'batch_size', 'max_len', 'seed', 'lr'):
            if previous['args'][key] != config['args'][key]:
                raise ValueError(f'Resume training setting changed: {key}')
    elif (out / 'config.json').exists():
        raise FileExistsError('Output run exists; use --resume or another --out')
    model = YLMCausalLM(cfg).to(device)
    if args.init_encoder:
        load_encoder_weights(model, args.init_encoder, log)
    # Persist config before training so an interrupted session can resume.
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    with Profiler(args.profile_dir or out / 'profile', device) as profiler:
        state = train_lm(model, splits['train'], tok, out_dir=str(out), tokens_budget=args.tokens_budget,
                         batch_size=args.batch_size, lr=args.lr, max_len=args.max_len, seed=args.seed,
                         device=device, ckpt_every=args.ckpt_every, resume=args.resume, profiler=profiler,
                         stop_after_steps=args.stop_after_steps, log=log)
    save_run_model(str(out), model, config)
    params = model.param_counts()
    metrics = {'validation': evaluate_lm(model, tok, splits['val'], args.max_len, args.batch_size, device),
               'numeric': numeric_eval(model, tok, splits['val'], args.numeric_eval_n, args.max_len, device),
               'params': params, 'tokens_seen': state['tokens'], 'token_budget': args.tokens_budget,
               'budget_shortfall': args.tokens_budget - state['tokens'], 'wall_seconds': state['wall'],
               'flops_estimate': 6 * params['total'] * state['tokens'],
               'flops_note': '6*N*T dense-training proxy; excludes attention quadratic and geometry graph/CG work',
               'state': state}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    log('saved', out / 'metrics.json')


if __name__ == '__main__':
    main()

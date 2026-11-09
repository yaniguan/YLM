"""Answer-only SFT or numeric preference DPO from a saved pretraining/SFT run."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.profiling import Profiler
from src.tasks.common import (load_run, train_lm, save_run_model, seed_everything, pick_device)
from src.tasks.qm9_pretrain import load_qm9
from src.tasks.sft import build_qa, build_preferences, train_dpo
from src.tasks.exposure import training_groups, merge_exposure, assert_no_group_leak
from src.tasks.evaluation import evaluate_lm, numeric_eval


def source_splits(config, sources, seed, limit, group_manifest=None):
    """Return QA split lists; builders never receive a combined train/held-out collection."""
    out = {k: [] for k in ('train', 'val', 'test')}
    manifests = {}
    pre = config.get('pretraining', config)
    args = pre['args']
    xyz = pre['args']['arm'] == 'smiles_xyz'
    for source in sources:
        if source == 'qm9':
            rows = load_qm9(args.get('qm9', 'data/qm9_xtb.pkl'), args.get('sdf', 'data/gdb9.sdf'), args.get('limit'))
            membership = {k: set(v) for k, v in pre['splits'].items()}
            splits = {k: [i for i, r in enumerate(rows) if r['qm9_index'] in ids] for k, ids in membership.items()}
        elif source == 'calisol':
            from src.tasks.calisol import load_calisol, split_calisol
            rows = load_calisol(limit=limit, seed=0)
            splits = split_calisol(rows, seed, 'group')
        else:
            from src.tasks.reddb import load_reddb, split_reddb
            rows = load_reddb(limit=limit, seed=0)
            splits = split_reddb(rows, seed, 'scaffold')
        if group_manifest is not None and source != "qm9":
            assert_no_group_leak(config, source, rows, splits)
            group_manifest[source] = training_groups(source, rows, splits["train"])
        manifests[source] = {k: [str(rows[i].get('id', rows[i].get('qm9_index'))) for i in ids]
                             for k, ids in splits.items()}
        for split, indices in splits.items():
            out[split] += build_qa([rows[i] for i in indices], xyz)
    return out, manifests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group_exposure_guard', action='store_true', help='Record training groups and check for overlap')
    p.add_argument('--pretrained', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--sources', nargs='+', choices=['qm9', 'calisol', 'reddb'], default=['qm9'])
    p.add_argument('--dpo', action='store_true')
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--tokens_budget', type=int, default=100000)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit', type=int)
    p.add_argument('--device')
    p.add_argument('--resume', action='store_true', help='Resume saved state if present; otherwise start fresh')
    p.add_argument('--ckpt_every', type=int, default=100)
    p.add_argument('--stop_after_steps', type=int)
    p.add_argument('--numeric_eval_n', type=int, default=32)
    p.add_argument('--profile_dir')
    args = p.parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok, source_config = load_run(args.pretrained, device)
    digest = hashlib.sha256()
    with open(Path(args.pretrained) / 'model.pt', 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    current_groups = {} if args.group_exposure_guard else None
    splits, manifests = source_splits(source_config, args.sources, args.seed, args.limit, current_groups)
    max_atoms = max((len(ex.geom['Z']) for examples in splits.values() for ex in examples
                     if ex.geom is not None), default=0)
    # Reserve geometry space in every arm so the question text cap remains comparable.
    effective_len = min(args.max_len, model.cfg.n_positions - max_atoms)
    if effective_len < 22:
        raise ValueError('Not enough LM context for questions and generation after reserving geometry')
    config = {'model_cfg': model.cfg.to_dict(), 'args': vars(args), 'qa_splits': manifests,
              'pretraining': source_config.get('pretraining', source_config),
              'effective_max_len': effective_len, 'initial_model_sha256': digest.hexdigest()}
    # Preserve training exposure through multiple SFT/DPO stages, even if sources change.
    exposure = {k: set(v) for k, v in source_config.get('sft_train_ids', {}).items()}
    for lineage in (source_config.get('qa_splits', {}), manifests):
        for source, membership in lineage.items():
            exposure.setdefault(source, set()).update(membership['train'])
    if args.group_exposure_guard or 'group_exposure' in source_config:
        # Mark new training groups as unknown when recording is turned off.
        lineage = dict(source_config)
        if not args.group_exposure_guard:
            lineage['qa_splits'] = {**source_config.get('qa_splits', {}), **manifests}
        config['group_exposure'] = merge_exposure(lineage, current_groups or {})
        if not args.group_exposure_guard:
            unknown = set(config['group_exposure']['unknown_sources'])
            for source in args.sources:
                if source in ('calisol', 'reddb'):
                    unknown.add(source)
            config['group_exposure']['unknown_sources'] = sorted(unknown)
    config['sft_train_ids'] = {k: sorted(v) for k, v in exposure.items()}
    if args.resume and (out / 'config.json').exists():
        old = json.loads((out / 'config.json').read_text())
        if old.get('group_exposure') != config.get('group_exposure'):
            raise ValueError('Resume group exposure changed')
        if old['qa_splits'] != manifests or old['model_cfg'] != config['model_cfg']:
            raise ValueError('Resume data/model changed')
        if old['initial_model_sha256'] != config['initial_model_sha256']:
            raise ValueError('Source checkpoint changed; the initial model/DPO reference must stay fixed')
        for k in ('pretrained', 'dpo', 'beta', 'seed', 'tokens_budget', 'batch_size', 'max_len', 'lr'):
            if old['args'][k] != vars(args)[k]:
                raise ValueError(f'Resume training setting changed: {k}')
    elif (out / 'config.json').exists():
        raise FileExistsError('Output exists; use --resume or another --out')
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    kwargs = dict(out_dir=str(out), tokens_budget=args.tokens_budget, batch_size=args.batch_size,
                  max_len=effective_len, lr=args.lr, seed=args.seed, device=device,
                  resume=args.resume, ckpt_every=args.ckpt_every, stop_after_steps=args.stop_after_steps)
    with Profiler(args.profile_dir or out / 'profile', device) as profiler:
        if args.dpo:
            reference, _, _ = load_run(args.pretrained, device)
            state = train_dpo(model, reference, build_preferences(splits['train'], args.seed), tok,
                              beta=args.beta, profiler=profiler, **kwargs)
        else:
            state = train_lm(model, splits['train'], tok, loss_kinds=['answer'], profiler=profiler, **kwargs)
    save_run_model(str(out), model, config)
    metrics = {'validation': evaluate_lm(model, tok, splits['val'], effective_len, args.batch_size,
                                        device, ['answer']),
               'numeric': numeric_eval(model, tok, splits['val'], args.numeric_eval_n,
                                       effective_len, device, answer_only=True),
               'state': state, 'budget_shortfall': args.tokens_budget - state['tokens'],
               'budget_note': 'DPO counts chosen+rejected policy AND reference forward tokens, including prefix',
               'params': model.param_counts()}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

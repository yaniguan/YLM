"""Frozen probes, full fine-tuning and public molecular fingerprint baselines."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.profiling import Profiler
from src.tasks.common import load_run, pick_device, seed_everything
from src.tasks.exposure import assert_no_group_leak
from src.tasks.downstream import (render_examples, controlled_examples, extract_features, ridge_predict, fingerprint_features,
                                  fine_tune, fine_tune_multitarget, result_record, paired_comparisons,
                                  mp_composition_features)


def load_task(task, limit):
    if task == 'calisol':
        from src.tasks.calisol import load_calisol, split_calisol
        return load_calisol(limit=limit, seed=0), split_calisol, 'group'
    if task == 'reddb':
        from src.tasks.reddb import load_reddb, split_reddb
        return load_reddb(limit=limit, seed=0), split_reddb, 'scaffold'
    from src.tasks.mp_battery import load_mp_battery, split_mp
    return load_mp_battery(limit=limit), split_mp, 'group'


def assert_no_sft_leak(config, task, rows, split):
    manifests = config.get('qa_splits', {}).get(task)
    trained = set(config.get('sft_train_ids', {}).get(task, []))
    if manifests:
        trained.update(manifests['train'])
    if trained:
        heldout = {str(rows[i]['id']) for k in ('val', 'test') for i in split[k]}
        if trained & heldout:
            raise ValueError('Downstream held-out rows overlap task-specific SFT train rows; use matching split seed/data')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry_control', choices=['original', 'zero', 'donor'], default='original')
    p.add_argument('--group_exposure_guard', action='store_true')
    p.add_argument('--multitarget', action='store_true', help='Train RedDB/MP targets together, ignoring missing labels')
    p.add_argument('--mp_composition_baseline', action='store_true',
                   help='Enable ridge regression using element fractions and the working ion')
    p.add_argument('--runs', nargs='+', required=True, help='Saved run directories, one per arm')
    p.add_argument('--tasks', nargs='+', choices=['calisol', 'reddb', 'mp_battery'], default=['calisol', 'reddb', 'mp_battery'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--split', choices=['random', 'group', 'doi', 'scaffold'])
    p.add_argument('--limit', type=int)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--probe_only', action='store_true')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_len', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--device')
    p.add_argument('--out', default='results/oct30_downstream')
    p.add_argument('--profile_dir')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.bootstrap < 1:
        p.error('--bootstrap must be positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    records = []
    run_configs = []
    for run in args.runs:
        config = json.loads((Path(run) / 'config.json').read_text())
        pre = config.get('pretraining', config)
        run_configs.append((run, config, pre['args']['arm']))
    if len({arm for _, _, arm in run_configs}) != len(run_configs):
        raise ValueError('Use one checkpoint per arm in each invocation')
    with Profiler(args.profile_dir or out / 'profile', device) as profiler:
        for task in args.tasks:
            rows, splitter, default_split = load_task(task, args.limit)
            if not rows:
                records.append({'task': task, 'skipped': 'No rows (MP requires MP_API_KEY)'})
                continue
            method_split = args.split or default_split
            # One common text cap across arms, reserving room for the longest geometry prefix.
            max_atoms = max(len(r['example'].geom['Z']) for r in rows)
            text_limit = min([args.max_len] + [c['model_cfg']['n_positions'] - max_atoms
                                              for _, c, _ in run_configs
                                              if c['model_cfg']['mode'] == 'prefix'])
            if text_limit < 2:
                raise ValueError('Geometry prefix exceeds saved LM context capacity')
            targets = sorted({k for r in rows for k in r['targets']})
            fp = fingerprint_features(rows) if task != 'mp_battery' else None
            composition = None
            if task == 'mp_battery' and args.mp_composition_baseline:
                composition = mp_composition_features(rows)
            for seed in args.seeds:
                seed_everything(seed)
                # Split ONCE before target filtering so missing labels cannot change membership.
                split = splitter(rows, seed, method_split)
                (out / f'{task}_split_seed{seed}.json').write_text(json.dumps(
                    {k: [rows[i]['id'] for i in v] for k, v in split.items()}, indent=2))
                valid = {target: {k: [i for i in ids if target in rows[i]['targets']] for k, ids in split.items()}
                         for target in targets}
                for target in targets:
                    if any(not valid[target][k] for k in ('train', 'val', 'test')):
                        records.append({'task': task, 'target': target, 'seed': seed,
                                        'skipped': 'Empty train/validation/test split after label filtering'})
                        continue
                    train, val, test = (valid[target][k] for k in ('train', 'val', 'test'))
                    y = np.array([r['targets'].get(target, np.nan) for r in rows])
                    if composition is not None:
                        pred, alpha = ridge_predict(composition, y, train, val, test)
                        records.append(dict(task=task, target=target, arm='composition', method='ridge', seed=seed,
                                            alpha=alpha, **result_record(rows, test, target, pred, seed, args.bootstrap)))
                    if fp is not None:
                        pred, alpha = ridge_predict(fp, y, train, val, test)
                        records.append(dict(task=task, target=target, arm='fingerprint', method='ridge', seed=seed,
                                            alpha=alpha, **result_record(rows, test, target, pred, seed, args.bootstrap)))
                        try:
                            from sklearn.ensemble import RandomForestRegressor
                        except ImportError:
                            records.append(dict(task=task, target=target, arm='fingerprint', method='RF', seed=seed,
                                                skipped='scikit-learn unavailable; closed-form ridge above remains available'))
                        else:
                            best = None
                            for leaf in (1, 3, 10):
                                rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=leaf,
                                                           random_state=seed, n_jobs=1)
                                rf.fit(fp[train], y[train])
                                loss = float(np.mean((rf.predict(fp[val]) - y[val]) ** 2))
                                if best is None or loss < best[0]:
                                    best = (loss, rf, leaf)
                            records.append(dict(task=task, target=target, arm='fingerprint', method='RF', seed=seed,
                                                min_samples_leaf=best[2], **result_record(
                                                    rows, test, target, best[1].predict(fp[test]), seed, args.bootstrap)))
                for run, config, arm in run_configs:
                    assert_no_sft_leak(config, task, rows, split)
                    if args.group_exposure_guard:
                        assert_no_group_leak(config, task, rows, split)
                    model, tok, _ = load_run(run, device)
                    examples, control = controlled_examples(
                        rows, arm == 'smiles_xyz', mode=args.geometry_control if arm != 'smiles' else 'original',
                        indices=split['test'], donor_indices=split['train'], seed=seed)
                    records.append(dict(task=task, arm=arm, seed=seed, geometry_control=control))
                    features = extract_features(model, tok, examples, device, args.batch_size, text_limit, profiler)
                    del model
                    joint = args.multitarget and task in ('reddb', 'mp_battery')
                    if joint and not args.probe_only:
                        eligible = [t for t in targets if all(valid[t][k] for k in ('train', 'val', 'test'))]
                        if eligible:
                            seed_everything(seed)
                            model, tok, _ = load_run(run, device)
                            y_joint = np.array([[r['targets'].get(t, np.nan) for t in eligible] for r in rows])
                            prediction = fine_tune_multitarget(
                                model, tok, examples, y_joint, split['train'], split['val'], split['test'],
                                seed=seed, epochs=args.epochs, device=device, batch_size=args.batch_size,
                                max_len=text_limit, lr=args.lr, profiler=profiler)
                            positions = {i: j for j, i in enumerate(split['test'])}
                            for column, target in enumerate(eligible):
                                test_ids = valid[target]['test']
                                pred = prediction[[positions[i] for i in test_ids], column]
                                records.append(dict(task=task, target=target, arm=arm, method='finetune_joint',
                                                    seed=seed, joint_targets=eligible,
                                                    **result_record(rows, test_ids, target, pred, seed, args.bootstrap)))
                            del model
                    for target in targets:
                        if any(not valid[target][k] for k in ('train', 'val', 'test')):
                            continue
                        train, val, test = (valid[target][k] for k in ('train', 'val', 'test'))
                        y = np.array([r['targets'].get(target, np.nan) for r in rows])
                        pred, alpha = ridge_predict(features, y, train, val, test)
                        records.append(dict(task=task, target=target, arm=arm, method='probe', seed=seed,
                                            alpha=alpha, **result_record(rows, test, target, pred, seed, args.bootstrap)))
                        if not args.probe_only and not joint:
                            seed_everything(seed)
                            model, tok, _ = load_run(run, device)
                            pred = fine_tune(model, tok, examples, y, train, val, test, seed=seed,
                                             epochs=args.epochs, device=device, batch_size=args.batch_size,
                                             max_len=text_limit, lr=args.lr, profiler=profiler)
                            records.append(dict(task=task, target=target, arm=arm, method='finetune', seed=seed,
                                                **result_record(rows, test, target, pred, seed, args.bootstrap)))
                            del model
                    # Persist after each arm so a long run has reviewable intermediate results.
                    (out / 'results.json').write_text(json.dumps({'args': vars(args), 'records': records}, indent=2))
    comparisons = paired_comparisons(records, args.bootstrap)
    aggregates = []
    keys = sorted({(r['task'], r['target'], r['arm'], r['method']) for r in records if 'metrics' in r})
    for task, target, arm, method in keys:
        subset = [r for r in records if (r.get('task'), r.get('target'), r.get('arm'), r.get('method')) == (task, target, arm, method)
                  and 'metrics' in r]
        aggregates.append({'task': task, 'target': target, 'arm': arm, 'method': method,
                           'seeds': [r['seed'] for r in subset], 'mean_metrics': {
                               k: float(np.mean(v)) if (v := [r['metrics'][k] for r in subset if r['metrics'][k] is not None]) else None
                               for k in ('mae', 'rmse', 'r2', 'spearman')}})
    result = {'args': vars(args), 'records': records, 'aggregates': aggregates, 'paired_comparisons': comparisons}
    (out / 'results.json').write_text(json.dumps(result, indent=2))
    lines = ['# Oct-30 downstream results', '',
             '| Task | Target | Arm | Method | Seeds | Mean MAE | Mean RMSE | Mean R2 | Mean Spearman |',
             '|---|---|---|---|---|---|---|---|---|']
    def number(v):
        return f'{v:.5g}' if v is not None else 'N/A'
    for r in aggregates:
        lines.append('| ' + ' | '.join([r['task'], r['target'], r['arm'], r['method'], str(len(r['seeds']))] +
                     [number(r['mean_metrics'][k]) for k in ('mae', 'rmse', 'r2', 'spearman')]) + ' |')
    lines += ['', 'Per-seed test bootstrap intervals, predictions, split IDs and paired comparisons are in results.json.',
              'Conductivity is evaluated on log10(k+0.001) and on the original mS/cm scale in results.json.',
              'MP has no Morgan fingerprint baseline because molecular fingerprints do not represent crystals.']
    for r in records:
        if 'skipped' in r:
            lines.append(f"\nSkipped {r.get('task')} {r.get('target', '')}: {r['skipped']}")
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()

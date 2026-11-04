"""Downstream probes, fine-tuning, fingerprint baselines and paired uncertainty."""
from __future__ import annotations
import math
import numpy as np
import torch
from src.tasks.common import Example, collate, model_inputs, xyz_segs


def rankdata(x):
    x = np.asarray(x)
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def regression_metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if not len(y):
        return {k: None for k in ('mae', 'rmse', 'r2', 'spearman')}
    residual = prediction - y
    denom = float(((y - y.mean()) ** 2).sum())
    ry, rp = rankdata(y), rankdata(prediction)
    corr_denom = np.linalg.norm(ry - ry.mean()) * np.linalg.norm(rp - rp.mean())
    return {'mae': float(np.abs(residual).mean()), 'rmse': float(np.sqrt((residual ** 2).mean())),
            'r2': float(1 - (residual ** 2).sum() / denom) if denom > 0 else None,
            'spearman': float(np.dot(ry - ry.mean(), rp - rp.mean()) / corr_denom) if corr_denom > 0 else None}


def bootstrap_metrics(y, prediction, seed=0, draws=1000):
    rng = np.random.default_rng(seed)
    values = {k: [] for k in regression_metrics(y, prediction)}
    y, prediction = np.asarray(y), np.asarray(prediction)
    for _ in range(draws):
        ids = rng.integers(0, len(y), len(y))
        for key, value in regression_metrics(y[ids], prediction[ids]).items():
            if value is not None and math.isfinite(value):
                values[key].append(value)
    return {k: np.percentile(v, [2.5, 97.5]).tolist() if v else None for k, v in values.items()}


def result_record(rows, indices, target, predictions, seed, draws=1000):
    y = np.array([rows[i]['targets'][target] for i in indices])
    predictions = np.asarray(predictions, dtype=float)
    result = {'metrics': regression_metrics(y, predictions),
              'bootstrap_95': bootstrap_metrics(y, predictions, seed, draws),
              'test_ids': [rows[i]['id'] for i in indices], 'truth': y.tolist(),
              'prediction': predictions.tolist(), 'n_test': len(indices)}
    if target == 'log_conductivity':
        from src.tasks.calisol import conductivity_from_log
        raw_y, raw_p = conductivity_from_log(y), conductivity_from_log(predictions)
        result['conductivity_mS_cm'] = {'metrics': regression_metrics(raw_y, raw_p),
                                      'bootstrap_95': bootstrap_metrics(raw_y, raw_p, seed, draws)}
    return result


def controlled_examples(rows, xyz=False, *, mode="original", indices=None, donor_indices=None, seed=0):
    from src.tasks.geometry_controls import geometry_control
    changed, report = geometry_control(rows, mode, indices=indices, donor_indices=donor_indices, seed=seed)
    return render_examples(changed, xyz), report


def render_examples(rows, xyz=False):
    examples = []
    for row in rows:
        ex = row['example']
        segs = list(ex.segs)
        if xyz and ex.geom is not None:
            segs += xyz_segs(ex.geom['Z'], ex.geom['pos'])
        examples.append(Example(segs, ex.geom, ex.meta))
    return examples


def pooled_hidden(model, batch):
    inputs = model_inputs(batch)
    inputs.pop('labels')
    hidden = model(**inputs, output_hidden=True)['hidden']
    mask = batch['attention_mask'].to(hidden.dtype)[..., None]
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)


@torch.no_grad()
def extract_features(model, tok, examples, device='cpu', batch_size=8, max_len=512, profiler=None):
    from contextlib import nullcontext
    model.eval()
    features = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        with profiler.phase('data') if profiler else nullcontext():
            batch = collate(chunk, tok, max_len, need_geom=model.needs_geometry, device=device)
        with profiler.phase('forward') if profiler else nullcontext():
            features.append(pooled_hidden(model, batch).cpu().numpy())
        if profiler:
            from src.tasks.common import batch_tokens
            profiler.step_done(batch_tokens(batch, model), len(chunk))
    return np.concatenate(features)


def ridge_predict(features, y, train, val, test, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    # Scaling and target centring use training rows only; alpha uses validation only.
    x = np.asarray(features, dtype=np.float64)
    mean, scale = x[train].mean(0), x[train].std(0)
    scale[scale < 1e-8] = 1
    x = (x - mean) / scale
    ymean = float(y[train].mean())
    xt = torch.as_tensor(x[train], dtype=torch.float64)
    yt = torch.as_tensor(y[train] - ymean, dtype=torch.float64)
    best = None
    for alpha in alphas:
        # Dual solve when the fingerprint dimension exceeds training set size.
        if xt.shape[1] > xt.shape[0]:
            gram = xt @ xt.T + alpha * torch.eye(len(train), dtype=xt.dtype)
            weight = xt.T @ torch.linalg.solve(gram, yt)
        else:
            gram = xt.T @ xt + alpha * torch.eye(xt.shape[1], dtype=xt.dtype)
            weight = torch.linalg.solve(gram, xt.T @ yt)
        w = weight.numpy()
        loss = float(np.mean((x[val] @ w + ymean - y[val]) ** 2))
        if best is None or loss < best[0]:
            best = (loss, w, alpha)
    return x[test] @ best[1] + ymean, best[2]


def fingerprint_features(rows):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import DataStructs
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    cached, vectors = {}, []
    for row in rows:
        vector = np.zeros(1024)
        for smiles, amount in row['fingerprints']:
            if smiles not in cached:
                mol = Chem.MolFromSmiles(smiles)
                arr = np.zeros(1024, dtype=np.float64)
                DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), arr)
                cached[smiles] = arr
            vector += amount * cached[smiles]
        conditions = list(row.get('conditions', []))
        if 'ratio_basis' in row:
            # Conditions include units/basis indicators so incompatible units are distinguishable.
            conditions += [float(row['ratio_basis'] == k) for k in ('mol', 'w', 'v')]
            conditions += [float(row['concentration_units'] == k) for k in ('mol/L', 'mol/kg')]
        vectors.append(np.r_[vector, conditions])
    return np.stack(vectors)


def fine_tune(model, tok, examples, y, train, val, test, *, seed=0, epochs=3, device='cpu',
              batch_size=8, max_len=512, lr=1e-5, profiler=None):
    from contextlib import nullcontext
    from src.tasks.common import batch_tokens
    # Caller reloads the original checkpoint for each target and seed.
    head = torch.nn.Linear(model.lm.config.hidden_size, 1).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr)
    centre, scale = float(y[train].mean()), max(float(y[train].std()), 1e-6)
    targets = torch.tensor((y - centre) / scale, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)

    @torch.no_grad()
    def predict(indices):
        model.eval()
        head.eval()
        values = []
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start + batch_size]
            batch = collate([examples[i] for i in ids], tok, max_len,
                            need_geom=model.needs_geometry, device=device)
            values.extend(head(pooled_hidden(model, batch)).squeeze(-1).cpu().tolist())
        return np.asarray(values) * scale + centre

    best_loss, best_model, best_head = float('inf'), None, None
    for _ in range(epochs):
        model.train()
        head.train()
        order = rng.permutation(train).tolist()
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            with profiler.phase('data') if profiler else nullcontext():
                batch = collate([examples[i] for i in ids], tok, max_len,
                                need_geom=model.needs_geometry, device=device)
            with profiler.phase('forward') if profiler else nullcontext():
                prediction = head(pooled_hidden(model, batch)).squeeze(-1)
                loss = torch.nn.functional.mse_loss(prediction, targets[ids])
            with profiler.phase('backward') if profiler else nullcontext():
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
            with profiler.phase('optimizer') if profiler else nullcontext():
                opt.step()
            if profiler:
                profiler.step_done(batch_tokens(batch, model), len(ids))
        value = float(np.mean((predict(val) - y[val]) ** 2))
        if value < best_loss:
            best_loss = value
            best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    if best_model is None:
        raise ValueError('Fine-tune requires positive epochs and finite validation loss')
    model.load_state_dict(best_model)
    head.load_state_dict(best_head)
    return predict(test)


def paired_comparisons(records, draws=1000, seed=0):
    """Hierarchical paired bootstrap: resample seeds, then shared test IDs within each seed."""
    rng = np.random.default_rng(seed)
    comparisons = []
    keys = sorted({(r['task'], r['target'], r['method']) for r in records if 'metrics' in r})
    for task, target, method in keys:
        candidates = [r for r in records if (r.get('task'), r.get('target'), r.get('method')) == (task, target, method)
                      and 'metrics' in r]
        reference = {r['seed']: r for r in candidates if r['arm'] == 'smiles'}
        for arm in sorted({r['arm'] for r in candidates} - {'smiles'}):
            other = {r['seed']: r for r in candidates if r['arm'] == arm}
            seeds = sorted(reference.keys() & other.keys())
            if not seeds:
                continue
            pairs = []
            for s in seeds:
                a, b = reference[s], other[s]
                if a['test_ids'] != b['test_ids'] or a['truth'] != b['truth']:
                    raise ValueError('Paired bootstrap requires identical test rows and labels')
                y = np.asarray(a['truth'])
                pairs.append(np.abs(np.asarray(b['prediction']) - y) - np.abs(np.asarray(a['prediction']) - y))
            values = []
            for _ in range(draws):
                sampled = rng.integers(0, len(pairs), len(pairs))
                values.append(float(np.mean([rng.choice(pairs[i], len(pairs[i]), replace=True).mean() for i in sampled])))
            comparisons.append({'task': task, 'target': target, 'method': method, 'arm': arm,
                                'reference': 'smiles', 'seeds': seeds,
                                'mae_delta': float(np.mean([p.mean() for p in pairs])),
                                'bootstrap_95': np.percentile(values, [2.5, 97.5]).tolist(),
                                'bootstrap_probability_delta_nonnegative': float(np.mean(np.asarray(values) >= 0))})
    return comparisons

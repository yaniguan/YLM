"""Numeric QA and synthetic DPO pairs. Call builders on explicit split subsets only."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from src.tasks.common import Example, Seg, fmt_num, molecule_segs


def build_qa(rows, with_xyz=False):
    questions = []
    for row in rows:
        source = row.get('example')
        geom = source.geom if source is not None else row.get('geom')
        context = (list(source.segs) if source is not None else
                   molecule_segs(row['smiles'], geom, with_xyz))
        if source is not None and with_xyz and geom is not None:
            from src.tasks.common import xyz_segs
            context += xyz_segs(geom['Z'], geom['pos'])
        for prop, value in row['targets'].items():
            if not np.isfinite(value):
                continue
            prompt = [Seg('Q: what is the ' + prop + ' of ')] + context + [Seg('? A: ')]
            questions.append(Example(prompt + [Seg(fmt_num(float(value), 6), 'answer', prop)], geom,
                                     {'target': float(value), 'property': prop,
                                      'source_id': row.get('id', row.get('qm9_index'))}))
    return questions


def question_prompt(ex):
    return Example([s for s in ex.segs if s.kind != 'answer'], ex.geom, ex.meta)


@dataclass
class PreferencePair:
    chosen: Example
    rejected: Example


def build_preferences(questions, seed=0, relative_noise=0.2):
    rng = np.random.default_rng(seed)
    pairs = []
    for ex in questions:
        value = ex.meta['target']
        delta = max(abs(value), 0.1) * relative_noise * rng.uniform(0.5, 1.5)
        wrong = value + delta * rng.choice([-1, 1])
        rejected = Example([Seg(fmt_num(wrong, 6), 'answer', s.prop) if s.kind == 'answer' else s
                            for s in ex.segs], ex.geom, dict(ex.meta))
        if rejected.text != ex.text:
            pairs.append(PreferencePair(ex, rejected))
    return pairs


def sequence_logps(model, batch):
    out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                labels=batch['labels'], geom=batch['geom'])
    # Sum only answer-token log probabilities; prompt tokens are masked by collate.
    return -out['nll'].sum(-1)


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.1):
    margin = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    return -F.logsigmoid(beta * margin).mean()


def train_dpo(model, reference, pairs, tok, *, out_dir, tokens_budget, batch_size=4, max_len=512,
              lr=1e-5, beta=0.1, seed=0, device='cpu', resume=False, ckpt_every=100,
              stop_after_steps=None, profiler=None):
    import time
    from src.tasks.common import (NullProfiler, collate, batch_tokens, epoch_order, lr_at,
                                  latest_checkpoint, load_checkpoint, save_checkpoint)
    if not pairs or tokens_budget < 1 or batch_size < 1:
        raise ValueError('DPO requires pairs and positive budget/batch size')
    profiler = profiler or NullProfiler()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    state = {'step': 0, 'tokens': 0, 'epoch': 0, 'pos': 0, 'wall': 0.0}
    ckpt_dir = str(out_dir) + '/ckpt'
    if resume:
        path = latest_checkpoint(ckpt_dir)
        if path is not None:
            state = load_checkpoint(path, model, opt)
    reference.eval().requires_grad_(False)
    model.train()
    start = time.perf_counter()
    session_steps = 0
    while state['tokens'] < tokens_budget:
        if stop_after_steps is not None and session_steps >= stop_after_steps:
            break
        if state['pos'] >= len(pairs):
            state['epoch'] += 1
            state['pos'] = 0
        order = epoch_order(len(pairs), seed, state['epoch'])
        selected = [pairs[i] for i in order[state['pos']:state['pos'] + batch_size]]
        with profiler.phase('data'):
            # Complete chosen/rejected pairs only. Include reference forward tokens in the budget.
            chosen = collate([p.chosen for p in selected], tok, max_len, ['answer'],
                             need_geom=model.needs_geometry, device=device)
            rejected = collate([p.rejected for p in selected], tok, max_len, ['answer'],
                               need_geom=model.needs_geometry, device=device)
            cost = 2 * (batch_tokens(chosen, model) + batch_tokens(rejected, model))
            while cost > tokens_budget - state['tokens'] and len(selected) > 1:
                selected = selected[:-1]
                chosen = collate([p.chosen for p in selected], tok, max_len, ['answer'],
                                 need_geom=model.needs_geometry, device=device)
                rejected = collate([p.rejected for p in selected], tok, max_len, ['answer'],
                                   need_geom=model.needs_geometry, device=device)
                cost = 2 * (batch_tokens(chosen, model) + batch_tokens(rejected, model))
        if cost > tokens_budget - state['tokens']:
            break
        if not bool((chosen['labels'][:, 1:] != -100).any(1).all()) or not bool(
                (rejected['labels'][:, 1:] != -100).any(1).all()):
            raise ValueError('DPO answers were truncated; increase --max_len')
        for g in opt.param_groups:
            g['lr'] = lr_at(state['tokens'] / tokens_budget, lr)
        with profiler.phase('forward'):
            with torch.no_grad():
                rc, rr = sequence_logps(reference, chosen), sequence_logps(reference, rejected)
            loss = dpo_loss(sequence_logps(model, chosen), sequence_logps(model, rejected), rc, rr, beta)
        with profiler.phase('backward'):
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        with profiler.phase('optimizer'):
            opt.step()
        state['step'] += 1
        state['tokens'] += cost
        state['pos'] += len(selected)
        state['loss'] = float(loss)
        session_steps += 1
        profiler.step_done(cost, len(selected))
        if ckpt_every and state['step'] % ckpt_every == 0:
            elapsed = state['wall'] + time.perf_counter() - start
            save_checkpoint(ckpt_dir, model, opt, dict(state, wall=elapsed))
    state['wall'] += time.perf_counter() - start
    save_checkpoint(ckpt_dir, model, opt, state)
    return state

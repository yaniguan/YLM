"""Evaluation shared by pretraining, post-training and downstream scripts."""
from __future__ import annotations
import math
import numpy as np
import torch
from src.tasks.common import (KINDS, Example, collate, model_inputs, greedy_decode, parse_number, encode_example)


@torch.no_grad()
def evaluate_lm(model, tok, examples, max_len=512, batch_size=8, device='cpu', loss_kinds=None):
    was_training = model.training
    model.eval()
    sums = np.zeros(len(KINDS))
    counts = np.zeros(len(KINDS), dtype=int)
    for start in range(0, len(examples), batch_size):
        batch = collate(examples[start:start + batch_size], tok, max_len, loss_kinds,
                        need_geom=model.needs_geometry, device=device)
        out = model(**model_inputs(batch))
        for i in range(len(KINDS)):
            keep = (batch['kinds'][:, 1:] == i) & (batch['labels'][:, 1:] != -100)
            sums[i] += float(out['nll'][keep].sum())
            counts[i] += int(keep.sum())
    model.train(was_training)
    loss = float(sums.sum() / counts.sum()) if counts.sum() else None
    return {'loss': loss, 'ppl': math.exp(min(loss, 80)) if loss is not None else None,
            'tokens': int(counts.sum()), 'by_token_type': {
                k: {'tokens': int(n), 'loss': float(s / n) if n else None,
                    'ppl': math.exp(min(float(s / n), 80)) if n else None}
                for k, s, n in zip(KINDS, sums, counts)}}


def numeric_eval(model, tok, examples, n=32, max_len=512, device='cpu', answer_only=False):
    prompts, targets, props = [], [], []
    attempted, truncated = {}, {}
    for ex in examples[:n]:
        for i, seg in enumerate(ex.segs):
            if seg.kind != ('answer' if answer_only else 'num') or not seg.prop:
                continue
            target = parse_number(seg.text)
            if target is None:
                continue
            attempted[seg.prop] = attempted.get(seg.prop, 0) + 1
            prompt = Example(ex.segs[:i], ex.geom)
            if len(encode_example(tok, prompt, 1000000, add_eos=False)[0]) > max_len - 20:
                truncated[seg.prop] = truncated.get(seg.prop, 0) + 1
                continue
            prompts.append(prompt)
            targets.append(target)
            props.append(seg.prop)
    outputs = greedy_decode(model, tok, prompts, max_new=20, max_len=max_len, device=device) if prompts else []
    results = {}
    for prop, count in sorted(attempted.items()):
        errors = []
        for p, target, output in zip(props, targets, outputs):
            if p != prop:
                continue
            value = parse_number(output)
            if value is not None:
                errors.append(abs(value - target))
        results[prop] = {'mae': float(np.mean(errors)) if errors else None,
                         'parsed': len(errors), 'attempted': count,
                         'truncated_prompts': truncated.get(prop, 0),
                         'parse_rate': len(errors) / count}
    return results

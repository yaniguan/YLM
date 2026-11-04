"""Optional force-question routing diagnostics; no model-output changes."""
from __future__ import annotations
import numpy as np
import torch


class RoutingAudit:
    """Aggregate each question once. Ranks use strict-greater (optimistic ties).

    Bonded-neighbor errors use the existing analysis distance proxy, 3.4 bohr,
    rather than asserting chemical bond provenance. Score scales are population
    standard deviations over valid atoms and valid question tokens.
    """
    def __init__(self, topk=(1, 3, 5), neighbor_cutoff=3.4):
        if any(k < 1 for k in topk):
            raise ValueError('top-k must be positive')
        self.topk = tuple(topk)
        self.neighbor_cutoff = neighbor_cutoff
        self.seen = set()
        self.rows = []

    @torch.no_grad()
    def update(self, question_ids, attention, target, task, atom_mask, pos,
               scores=None, text_mask=None):
        attention = attention.detach().cpu()
        target, task = torch.as_tensor(target).cpu(), torch.as_tensor(task).cpu()
        atom_mask, pos = atom_mask.cpu().bool(), pos.cpu()
        for b, identity in enumerate(question_ids):
            if int(task[b]) != 1 or identity in self.seen:
                continue
            t = int(target[b])
            if t < 0 or t >= atom_mask.shape[1] or not atom_mask[b, t]:
                raise ValueError('Force target must be an unmasked atom')
            p = attention[b].masked_fill(~atom_mask[b], 0)
            if not torch.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
                raise ValueError('Attention must have finite nonnegative mass')
            p = p / p.sum()
            peak = int(p.masked_fill(~atom_mask[b], -1).argmax())
            rank = 1 + int(((p > p[t]) & atom_mask[b]).sum())
            exact = peak == t
            near = (pos[b, peak] - pos[b, t]).norm().item() < self.neighbor_cutoff
            row = {'entropy': float(-(p * p.clamp_min(1e-30).log()).sum()),
                   'target_rank': rank, 'target_mass': float(p[t]),
                   'exact': float(exact), 'neighbor_error': float(not exact and near),
                   'other_error': float(not exact and not near)}
            for k in self.topk:
                row[f'top{k}'] = float(rank <= min(k, int(atom_mask[b].sum())))
            if scores is not None:
                tm = (text_mask[b].cpu().bool() if text_mask is not None
                      else torch.ones(scores['invariant'].shape[1], dtype=torch.bool))
                valid = tm[:, None] & atom_mask[b][None, :]
                for name, values in scores.items():
                    selected = values[b].detach().cpu()[valid]
                    row[name + '_scale'] = float(selected.std(unbiased=False)) if len(selected) else 0.
            self.rows.append(row)
            self.seen.add(identity)

    def report(self):
        return {'n_questions': len(self.rows), 'rank_ties': 'strict-greater',
                'neighbor_cutoff_bohr': self.neighbor_cutoff,
                **{k: float(np.mean([r[k] for r in self.rows if k in r]))
                   for k in sorted({k for r in self.rows for k in r})}}

"""Optional auxiliary loss that supervises the pooled cross-attention to peak on the asked atom.

Motivation: Option B misses the atom-accuracy target (TEST atom acc 0.696 vs >= 0.754, README A/B/C table),
while its force MAE/cosine pass. atom_acc is argmax(attn_pool) == asked atom on force questions, so a small
cross-entropy term on attn_pool for force questions targets that metric directly. Off by default (weight 0).
"""
import torch


def attention_supervision_loss(attn_pool, atom_idx, task, atom_mask=None, force_task=1, eps=1e-8):
    """Mean -log attn_pool[b, atom_idx[b]] over force questions in the batch (0 if there are none).

    attn_pool: (B, N) non-negative attention over atoms (rows sum to ~1 over real atoms).
    atom_idx: (B,) asked atom index (ignored for non-force rows). task: (B,) task ids.
    atom_mask: optional (B, N) bool; padded atoms are excluded and rows renormalized.
    """
    sel = task == force_task
    if not bool(sel.any()):
        return attn_pool.sum() * 0.0
    a = attn_pool[sel].clamp_min(0)
    if atom_mask is not None:
        a = a * atom_mask[sel].to(a.dtype)
    a = a / a.sum(-1, keepdim=True).clamp_min(eps)
    p = a.gather(1, atom_idx[sel].long().view(-1, 1)).squeeze(1)
    return -(p + eps).log().mean()


def atom_hit_rate(attn_pool, atom_idx, task, force_task=1):
    """Fraction of force questions whose attention argmax is the asked atom (same as 04_train atom_acc)."""
    sel = task == force_task
    if not bool(sel.any()):
        return float("nan")
    return float((attn_pool[sel].argmax(-1) == atom_idx[sel]).float().mean())

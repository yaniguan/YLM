"""Padding for precomputed question embeddings; no tokenizer downloads required."""
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_molecules(samples):
    """Batch dictionaries with Z (N,), pos (N,3), text (T,d), optional task.

    Inputs must share device and floating dtype. Every molecule/question must be
    nonempty. Returns keyword arguments accepted by YLM.forward.
    """
    if not samples:
        raise ValueError("samples must be nonempty")
    for s in samples:
        if s['Z'].ndim != 1 or s['Z'].numel() == 0:
            raise ValueError("Z must contain at least one atom")
        if s['pos'].shape != (len(s['Z']), 3):
            raise ValueError("pos must have shape (N, 3)")
        if s['text'].ndim != 2 or len(s['text']) == 0:
            raise ValueError("text must have shape (T, d) with T > 0")
    batch = {k: pad_sequence([s[k] for s in samples], batch_first=True)
             for k in ('Z', 'pos', 'text')}
    for key, source in [('atom_mask', 'Z'), ('text_mask', 'text')]:
        batch[key] = pad_sequence([torch.ones(len(s[source]), dtype=torch.bool,
                                             device=s[source].device) for s in samples],
                                  batch_first=True)
    batch['task'] = torch.tensor([s.get('task', 0) for s in samples],
                                 dtype=torch.long, device=batch['Z'].device)
    return batch

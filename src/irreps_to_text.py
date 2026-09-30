"""Irreps-to-Text cross-attention, Option A (see 03_technical_design.md).

Text tokens ask, atoms answer.
  query   = from text hidden state h_t          (invariant)
  key     = from atom invariants                (invariant: scalars + norm of every l>=1 channel)
  value   = o3.Linear(atom irreps)              (equivariant)
  weight  = softmax over atoms of query . key   (invariant)
  out_t   = gate_attn(h_t) * sum_i weight_ti value_i  +  gate_sum(h_t) * sum_i value_i

The second term lets the text ask for a whole-molecule sum (needed for the dipole).
Gates are invariant numbers per channel, so out_t is equivariant.
"""
import math

import torch
import torch.nn as nn
from e3nn import o3


def channel_index(irreps):
    """For every position in the feature vector, which channel (irrep copy) it belongs to."""
    idx = []
    c = 0
    for mul, ir in irreps:
        for _ in range(mul):
            idx += [c] * ir.dim
            c += 1
    return torch.tensor(idx, dtype=torch.long)


def invariants(x, irreps):
    """Scalars as they are; for l>=1, the length of each channel. (..., dim) -> (..., n_inv)"""
    parts = []
    for (mul, ir), sl in zip(irreps, irreps.slices()):
        block = x[..., sl]
        if ir.l == 0:
            parts.append(block)
        else:
            block = block.reshape(*block.shape[:-1], mul, ir.dim)
            parts.append((block.pow(2).sum(-1) + 1e-8).sqrt())
    return torch.cat(parts, dim=-1)


class IrrepsToText(nn.Module):
    def __init__(self, text_dim, atom_irreps, out_irreps, key_dim=64):
        super().__init__()
        self.atom_irreps = o3.Irreps(atom_irreps)
        self.out_irreps = o3.Irreps(out_irreps)
        n_inv = self.atom_irreps.num_irreps
        self.key_dim = key_dim
        self.q = nn.Linear(text_dim, key_dim)
        self.k = nn.Sequential(nn.Linear(n_inv, 128), nn.SiLU(), nn.Linear(128, key_dim))
        self.v = o3.Linear(self.atom_irreps, self.out_irreps)
        self.gate_attn = nn.Linear(text_dim, self.out_irreps.num_irreps)
        self.gate_sum = nn.Linear(text_dim, self.out_irreps.num_irreps)
        self.register_buffer("expand", channel_index(self.out_irreps), persistent=False)

    def forward(self, h, x, atom_mask):
        """h: (B, T, text_dim)  x: (B, N, atom_irreps.dim)  atom_mask: (B, N)
        returns e: (B, T, out_irreps.dim), attn: (B, T, N)"""
        q = self.q(h)                                                  # (B, T, K)
        k = self.k(invariants(x, self.atom_irreps))                    # (B, N, K)
        scores = torch.einsum("btk,bnk->btn", q, k) / math.sqrt(self.key_dim)
        scores = scores.masked_fill(~atom_mask[:, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)                                  # (B, T, N)  invariant

        v = self.v(x) * atom_mask[..., None]                           # (B, N, D)  equivariant
        picked = torch.einsum("btn,bnd->btd", attn, v)                 # weighted average
        total = v.sum(dim=1, keepdim=True)                             # whole-molecule sum

        g_attn = torch.sigmoid(self.gate_attn(h))[..., self.expand]   # (B, T, D) invariant
        g_sum = self.gate_sum(h)[..., self.expand]
        e = g_attn * picked + g_sum * total
        return e, attn

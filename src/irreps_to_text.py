"""Irreps-to-Text cross-attention, Options A/B/C (03_technical_design.md).

Text tokens ask, atoms answer.
  query   = from text hidden state h_t          (invariant)
  key     = from atom invariants                (invariant: scalars + norm of every l>=1 channel)
  value   = o3.Linear(atom irreps)              (equivariant)
  weight  = softmax over atoms of query . key   (invariant)
  out_t   = gate_attn(h_t) * sum_i weight_ti value_i  +  gate_sum(h_t) * sum_i value_i

The second term lets the text ask for a whole-molecule sum (needed for the dipole).
Gates are invariant numbers per channel, so out_t is equivariant.
B adds text-gated global/atom same-irrep contractions to scores.
C applies a per-token text-weighted CG self product to A's output.
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


def invariants(x, irreps, legacy_odd_scalars=False):
    """Parity-even keys; legacy_odd_scalars reproduces the SO(3)-only ablation.

    Even scalars unchanged, odd scalars absolute, higher-l channel lengths.
    (..., dim) -> (..., n_inv).
    """
    parts = []
    for (mul, ir), sl in zip(irreps, irreps.slices()):
        block = x[..., sl]
        if ir.l == 0:
            parts.append(block if ir.p == 1 or legacy_odd_scalars else block.abs())
        else:
            block = block.reshape(*block.shape[:-1], mul, ir.dim)
            parts.append((block.pow(2).sum(-1) + 1e-8).sqrt())
    return torch.cat(parts, dim=-1)


class IrrepsToText(nn.Module):
    def __init__(self, text_dim, atom_irreps, out_irreps, key_dim=64, option="A", legacy_odd_scalars=False, b_score_mode="legacy"):
        super().__init__()
        if option not in {"A", "B", "C"}:
            raise ValueError("option must be A, B, or C")
        if b_score_mode not in {"legacy", "normalized"}:
            raise ValueError("Unknown B score mode")
        self.b_score_mode = b_score_mode
        self.audit_enabled = False
        self.last_scores = None
        self.option = option
        self.legacy_odd_scalars = legacy_odd_scalars
        self.atom_irreps = o3.Irreps(atom_irreps)
        self.out_irreps = o3.Irreps(out_irreps)
        n_inv = self.atom_irreps.num_irreps
        self.key_dim = key_dim
        self.q = nn.Linear(text_dim, key_dim)
        self.k = nn.Sequential(nn.Linear(n_inv, 128), nn.SiLU(), nn.Linear(128, key_dim))
        # B contracts identical (l, parity) channels, preserving O(3).
        if option == "B":
            if b_score_mode == "normalized":
                self.direction_coefficient = nn.Parameter(torch.zeros(()))
            self.direction_key = o3.Linear(self.atom_irreps, self.atom_irreps)
            self.direction_query = nn.Linear(text_dim, self.atom_irreps.num_irreps)
        if option == "C":
            # C uses A's output and text-generated weights for every CG path.
            self.tp = o3.FullyConnectedTensorProduct(
                self.out_irreps, self.out_irreps, self.out_irreps,
                internal_weights=False, shared_weights=False)
            self.tp_weights = nn.Sequential(nn.Linear(text_dim, 64), nn.SiLU(),
                                            nn.Linear(64, self.tp.weight_numel))
        self.v = o3.Linear(self.atom_irreps, self.out_irreps)
        self.gate_attn = nn.Linear(text_dim, self.out_irreps.num_irreps)
        self.gate_sum = nn.Linear(text_dim, self.out_irreps.num_irreps)
        self.register_buffer("expand", channel_index(self.out_irreps), persistent=False)

    def forward(self, h, x, atom_mask):
        """h: (B, T, text_dim)  x: (B, N, atom_irreps.dim)  atom_mask: (B, N)
        returns e: (B, T, out_irreps.dim), attn: (B, T, N)"""
        q = self.q(h)                                                  # (B, T, K)
        k = self.k(invariants(x, self.atom_irreps, self.legacy_odd_scalars))                    # (B, N, K)
        scores = torch.einsum("btk,bnk->btn", q, k) / math.sqrt(self.key_dim)
        invariant_scores = scores
        if self.option == "B":
            global_x = (x * atom_mask[..., None]).sum(1) / atom_mask.sum(1, keepdim=True)
            gates = self.direction_query(h)
            keys = self.direction_key(x)
            channel = 0
            for (mul, ir), sl in zip(self.atom_irreps, self.atom_irreps.slices()):
                query = global_x[:, None, sl].reshape(x.shape[0], 1, mul, ir.dim)
                query = query * gates[..., channel:channel + mul, None]
                key = keys[..., sl].reshape(x.shape[0], x.shape[1], mul, ir.dim)
                if self.b_score_mode == "normalized":
                    raw = global_x[:, None, sl].reshape(x.shape[0], 1, mul, ir.dim)
                    query = torch.nn.functional.normalize(raw, dim=-1, eps=1e-6)
                    query = query * gates[..., channel:channel + mul, None]
                    key = torch.nn.functional.normalize(key, dim=-1, eps=1e-6)
                    contribution = torch.einsum("btcm,bncm->btn", query, key)
                    contribution = self.direction_coefficient.tanh() * contribution / math.sqrt(self.atom_irreps.num_irreps)
                    scores = scores + contribution
                else:
                    scores = scores + torch.einsum("btcm,bncm->btn", query, key)
                channel += mul
        if self.audit_enabled:
            self.last_scores = {
                "invariant": invariant_scores.detach(),
                "directional": (scores - invariant_scores).detach(),
            }
        else:
            self.last_scores = None
        scores = scores.masked_fill(~atom_mask[:, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)                                  # (B, T, N)  invariant

        v = self.v(x) * atom_mask[..., None]                           # (B, N, D)  equivariant
        picked = torch.einsum("btn,bnd->btd", attn, v)                 # weighted average
        total = v.sum(dim=1, keepdim=True)                             # whole-molecule sum

        g_attn = torch.sigmoid(self.gate_attn(h))[..., self.expand]   # (B, T, D) invariant
        g_sum = self.gate_sum(h)[..., self.expand]
        e = g_attn * picked + g_sum * total
        if self.option == "C":
            e = self.tp(e, e, self.tp_weights(h))
        return e, attn

"""YLM with configurable A/B/C cross-attention.

  question tokens (frozen text encoder, precomputed) ─┐
                                                      ├─► IrrepsToText ─► e_t per token
  atoms ─► Encoder (equivariant) ─► irreps per atom ──┘
                                                          │
            pool tokens with invariant weights  ◄─────────┘
                                                          │
            o3.Linear -> 1x1o  (one vector: dipole or force)

option: A invariant keys; B direction-aware keys; C text-weighted CG product.
legacy_odd_scalars: reproduce raw 0o keys for parity ablations (SO(3) only).

text_mode:
  "text"   : use the precomputed token states of the question (MiniLM / MatSciBERT)
  "taskid" : baseline v. Throw the question away, use a learned token for the task
             ("dipole" or "force"). The model then cannot know WHICH atom is asked about.
"""
import math

import torch
import torch.nn as nn
from e3nn import o3

from src.encoder import Encoder
from src.irreps_to_text import IrrepsToText, invariants


class _TwoStreamStep(nn.Module):
    """One interactive YLM block: e_t invariants update h_t, then h_t re-queries atom irreps."""

    def __init__(self, d, atom_irreps, cross_irreps, option="A", legacy_odd_scalars=False,
                 b_score_mode="legacy", n_heads=4):
        super().__init__()
        self.cross_irreps = o3.Irreps(cross_irreps)
        self.legacy_odd_scalars = legacy_odd_scalars
        self.n_heads = n_heads
        n_inv = self.cross_irreps.num_irreps
        self.norm_h = nn.LayerNorm(d)
        self.norm_e = nn.LayerNorm(n_inv)
        self.inv_mlp = nn.Sequential(nn.Linear(d + n_inv, d), nn.SiLU(), nn.Linear(d, d))
        self.norm_sa = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out_sa = nn.Linear(d, d)
        self.cross = IrrepsToText(d, atom_irreps, cross_irreps, option=option,
                                  legacy_odd_scalars=legacy_odd_scalars, b_score_mode=b_score_mode)
        self.mix_e = o3.Linear(cross_irreps, cross_irreps)

    def forward(self, h, e, x, atom_mask, text_mask):
        B, T, d = h.shape
        tmask = text_mask[..., None].to(h.dtype)
        h = h * tmask
        inv_e = invariants(e, self.cross_irreps, self.legacy_odd_scalars)
        h = (h + self.inv_mlp(torch.cat([self.norm_h(h), self.norm_e(inv_e)], dim=-1))) * tmask

        qkv = (self.qkv(self.norm_sa(h)) * tmask).reshape(B, T, 3, self.n_heads, d // self.n_heads)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(d // self.n_heads)
        has_tok = text_mask.any(dim=-1, keepdim=True)
        safe_mask = text_mask.clone()
        safe_mask[~has_tok.squeeze(-1), 0] = True
        scores = scores.masked_fill(~safe_mask[:, None, None, :], float("-inf"))
        sa = torch.einsum("bhts,bhsd->bhtd", scores.softmax(dim=-1), v).transpose(1, 2).reshape(B, T, d)
        h = (h + self.out_sa(sa) * has_tok[..., None].to(h.dtype)) * tmask

        e_new, attn_new = self.cross(h, x, atom_mask)
        return h, self.mix_e(e) + e_new, attn_new


class YLM(nn.Module):
    def __init__(self, text_dim, text_mode="text", hidden_irreps="64x0e + 8x0o + 32x1o + 8x1e + 16x2e",
                 cross_irreps="32x0e + 32x1o + 8x1e + 8x2e", n_layers=3, n_tasks=2, option="A",
                 legacy_odd_scalars=False, b_score_mode="legacy", n_cross_layers=1):
        super().__init__()
        if n_cross_layers < 1:
            raise ValueError("n_cross_layers must be >= 1")
        self.text_mode = text_mode
        self.n_cross_layers = n_cross_layers
        self.encoder = Encoder(hidden_irreps=hidden_irreps, n_layers=n_layers)
        self.task_embed = nn.Embedding(n_tasks, text_dim)             # only used in "taskid"
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, 256), nn.SiLU())
        self.cross = IrrepsToText(256, self.encoder.irreps_out, cross_irreps, option=option,
                                  legacy_odd_scalars=legacy_odd_scalars, b_score_mode=b_score_mode)
        if n_cross_layers > 1:
            self.extra_steps = nn.ModuleList([
                _TwoStreamStep(256, self.encoder.irreps_out, cross_irreps, option=option,
                               legacy_odd_scalars=legacy_odd_scalars, b_score_mode=b_score_mode)
                for _ in range(n_cross_layers - 1)
            ])
        self.token_weight = nn.Linear(256, 1)
        self.readout = o3.Linear(cross_irreps, "1x1o")

    def forward(self, Z, pos, atom_mask, text, text_mask, task):
        if self.text_mode == "taskid":
            text = self.task_embed(task)[:, None, :]                  # (B, 1, d)
            text_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
        h = self.text_proj(text)                                      # (B, T, 256) invariant
        x = self.encoder(Z, pos, atom_mask)                           # (B, N, D)   equivariant
        e, attn = self.cross(h, x, atom_mask)                         # (B, T, D')  equivariant
        if self.n_cross_layers > 1:
            attn_sum = attn
            for step in self.extra_steps:
                h, e, attn_step = step(h, e, x, atom_mask, text_mask)
                attn_sum = attn_sum + attn_step
            attn = attn_sum / float(self.n_cross_layers)

        w = self.token_weight(h).squeeze(-1).masked_fill(~text_mask, float("-inf")).softmax(-1)
        e_pool = torch.einsum("bt,btd->bd", w, e)                     # (B, D')
        attn_pool = torch.einsum("bt,btn->bn", w, attn)               # (B, N)  for analysis
        return self.readout(e_pool), attn_pool                        # (B, 3)

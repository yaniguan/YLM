"""Option B/C-lite: two text-chosen "pointers" plus an invariant-weighted combination of vector terms.

Each of K=2 heads pools the text with its own token weights, asks Option-A style attention for an atom,
and returns
    F_k = sum_i a_ki V(x_i)            equivariant l=1 value      ("the force there")
    P_k = sum_i a_ki (r_i - centroid)  equivariant l=1 position   ("where it is")
The pointer direction is u = unit(P_2 - P_1). The answer is a sum of vector terms with
INVARIANT coefficients c(text):
    c1 F_1 + c2 F_2 + c3 (F_1.u)u + c4 (F_2.u)u
Dot products and u(u.F) are CG products 1x1->0 and 0x1->1, so the output is exactly equivariant.
  diff  ("F_A - F_B")           : c = (+1, -1, 0, 0)
  proj  ("F_A along A->B")      : c = (0, 0, 1, 0)
Option A (one convex combination of atom vectors) is NOT provably unable to build these: with perfect per-atom forces
the closest convex combination already reaches cosine ~0.97 (diff) / ~0.99 (proj) on TensorQA-v1
(scripts/28_convex_ceiling.py). The gap is about what is learnable from the text and invariants, not expressivity.
"""
import math

import torch
import torch.nn as nn
from e3nn import o3

from src.encoder import Encoder
from src.irreps_to_text import invariants


class CompModel(nn.Module):
    def __init__(self, text_dim, hidden_irreps="64x0e + 8x0o + 32x1o + 8x1e + 16x2e", n_layers=3, d=256, key=64,
                 n_heads=2, token_dropout=0.0, parity="so3"):
        super().__init__()
        self.parity = parity
        self.token_dropout = token_dropout
        self.encoder = Encoder(hidden_irreps=hidden_irreps, n_layers=n_layers)
        irreps = self.encoder.irreps_out
        self.n_heads, self.key = n_heads, key
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d), nn.SiLU())
        self.token_w = nn.ModuleList([nn.Linear(d, 1) for _ in range(n_heads)])
        self.q = nn.ModuleList([nn.Linear(d, key) for _ in range(n_heads)])
        self.k = nn.Sequential(nn.Linear(irreps.num_irreps, 128), nn.SiLU(), nn.Linear(128, key))
        self.value = o3.Linear(irreps, "1x1o")
        self.coef = nn.Sequential(nn.Linear(n_heads * d, d), nn.SiLU(), nn.Linear(d, 2 * n_heads))

    def forward(self, Z, pos, atom_mask, text, text_mask):
        if self.training and self.token_dropout > 0:     # wording robustness: drop random tokens, keep [CLS]
            keep = torch.rand(text_mask.shape, device=text_mask.device) > self.token_dropout
            keep[:, 0] = True
            text_mask = text_mask & keep
        h = self.text_proj(text.to(self.text_proj[1].weight.dtype))                                            # (B, T, d) invariant
        x = self.encoder(Z, pos, atom_mask)
        keys = self.k(invariants(x, self.encoder.irreps_out, self.parity))               # (B, N, key)
        v = self.value(x) * atom_mask[..., None]                            # (B, N, 3) equivariant
        centered = (pos - (pos * atom_mask[..., None]).sum(1, keepdim=True)
                    / atom_mask.sum(1)[:, None, None]) * atom_mask[..., None]
        F, P, pooled, attns = [], [], [], []
        for k in range(self.n_heads):
            w = self.token_w[k](h).squeeze(-1).masked_fill(~text_mask, float("-inf")).softmax(-1)
            hk = torch.einsum("bt,btd->bd", w, h)
            a = (torch.einsum("bk,bnk->bn", self.q[k](hk), keys) / math.sqrt(self.key))
            a = a.masked_fill(~atom_mask, float("-inf")).softmax(-1)        # (B, N) invariant
            F.append(torch.einsum("bn,bnc->bc", a, v))
            P.append(torch.einsum("bn,bnc->bc", a, centered))
            pooled.append(hk)
            attns.append(a)
        u = P[1] - P[0]
        u = u / (u.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()              # unit vector, equivariant
        terms = F + [(f * u).sum(-1, keepdim=True) * u for f in F]          # 2K vector terms
        c = self.coef(torch.cat(pooled, -1))                                # (B, 2K) invariant
        y = sum(c[:, t, None] * terms[t] for t in range(len(terms)))
        self.last = dict(attn=[a.detach() for a in attns], c=c.detach())     # for scripts/14 diagnostics
        return y, attns[0]

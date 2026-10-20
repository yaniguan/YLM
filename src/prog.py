"""ProgModel: a LEARNED parser with an EXACT executor.

The text is turned into a small program by learned heads:
    task  in {force, diff, proj}   (classification head on the pooled question)
    ptr1, ptr2                     (two pointer heads over atoms, as in Option B/C-lite: pooled text token weights ->
                                    query -> attention over invariant atom keys)
and the program is run by fixed equivariant operators on a per-atom vector field V (the "physical quantity"):
    force: V_1      diff: V_1 - V_2      proj: (V_1 . u) u, u = unit(P_2 - P_1)
where V_k and P_k are the attention-weighted vector / centred position of pointer k.
Training mixes the three operators with softmax task probabilities and uses soft pointers; in eval mode the task
and pointers are hard (argmax). The program heads can be supervised with program labels (task, atom A, atom B),
which is extra supervision compared with Option B/C-lite; the training script reports that explicitly.

Output is in physical units (the training script divides by the task scale). Exactly equivariant (O(3) with
parity="o3"): only invariant weights, sums and dot products of equivariant vectors.
"""
import math

import torch
import torch.nn as nn
from e3nn import o3

from src.encoder import Encoder
from src.irreps_to_text import invariants

OPS = ("force", "diff", "proj")


class ProgModel(nn.Module):
    def __init__(self, text_dim, hidden_irreps="64x0e + 8x0o + 32x1o + 8x1e + 16x2e", n_layers=3, d=256, key=64,
                 parity="o3"):
        super().__init__()
        self.parity, self.key = parity, key
        self.encoder = Encoder(hidden_irreps=hidden_irreps, n_layers=n_layers)
        irreps = self.encoder.irreps_out
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d), nn.SiLU())
        self.token_w = nn.ModuleList([nn.Linear(d, 1) for _ in range(2)])
        self.q = nn.ModuleList([nn.Linear(d, key) for _ in range(2)])
        self.k = nn.Sequential(nn.Linear(irreps.num_irreps, 128), nn.SiLU(), nn.Linear(128, key))
        self.value = o3.Linear(irreps, "1x1o")
        self.task_head = nn.Sequential(nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, len(OPS)))

    def forward(self, Z, pos, atom_mask, text, text_mask):
        h = self.text_proj(text.to(pos.dtype))
        x = self.encoder(Z, pos, atom_mask)
        keys = self.k(invariants(x, self.encoder.irreps_out, self.parity))
        v = self.value(x) * atom_mask[..., None]                              # (B, N, 3) equivariant
        self.last_V = v
        centered = (pos - (pos * atom_mask[..., None]).sum(1, keepdim=True)
                    / atom_mask.sum(1)[:, None, None]) * atom_mask[..., None]
        F, P, pooled, logits = [], [], [], []
        for k in range(2):
            w = self.token_w[k](h).squeeze(-1).masked_fill(~text_mask, float("-inf")).softmax(-1)
            hk = torch.einsum("bt,btd->bd", w, h)
            lg = torch.einsum("bk,bnk->bn", self.q[k](hk), keys) / math.sqrt(self.key)
            lg = lg.masked_fill(~atom_mask, float("-inf"))
            a = lg.softmax(-1)
            if not self.training:                                              # hard pointers at eval
                a = torch.nn.functional.one_hot(lg.argmax(-1), lg.shape[-1]).to(lg.dtype)
            F.append(torch.einsum("bn,bnc->bc", a, v))
            P.append(torch.einsum("bn,bnc->bc", a, centered))
            pooled.append(hk)
            logits.append(lg)
        cls = (h * text_mask[..., None]).sum(1) / text_mask.sum(1, keepdim=True)
        task_logits = self.task_head(torch.cat(pooled + [cls], -1))
        p = task_logits.softmax(-1)
        if not self.training:
            p = torch.nn.functional.one_hot(task_logits.argmax(-1), len(OPS)).to(p.dtype)
        u = P[1] - P[0]
        u = u / (u.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        y = p[:, 0:1] * F[0] + p[:, 1:2] * (F[0] - F[1]) + p[:, 2:3] * (F[0] * u).sum(-1, keepdim=True) * u
        return y, dict(task_logits=task_logits, ptr_logits=logits)

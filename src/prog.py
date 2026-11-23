"""ProgModel: learned question parser paired with an exact equivariant executor.

The text is mapped to a discrete program by learned heads:
    task in {force, diff, proj}   (classifier on pooled question features)
    ptr1, ptr2                    (two invariant-key atom pointer heads)
and executed by fixed equivariant operators on a per-atom vector field V:
    force: V_1      diff: V_1 - V_2      proj: (V_1 . u) u, u = unit(P_2 - P_1)
where V_k and P_k are the pointer-weighted vector and centered position of head k.

Training mixes the three operators with softmax task probabilities and soft pointers;
eval mode switches to hard argmax task and pointer selection. Output is in physical
units and is O(3)-equivariant when parity="o3" (odd scalar sign dropped in keys).
"""
import math

import torch
import torch.nn as nn
from e3nn import o3

from src.encoder import Encoder
from src.irreps_to_text import invariants

OPS = ("force", "diff", "proj")


class ProgModel(nn.Module):
    def __init__(
        self,
        text_dim,
        hidden_irreps="64x0e + 8x0o + 32x1o + 8x1e + 16x2e",
        n_layers=3,
        d=256,
        key=64,
        parity="o3",
    ):
        super().__init__()
        if parity not in {"o3", "so3"}:
            raise ValueError("parity must be 'o3' or 'so3'")
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
        keys = self.k(invariants(x, self.encoder.irreps_out, legacy_odd_scalars=(self.parity == "so3")))
        v = self.value(x) * atom_mask[..., None]
        self.last_V = v
        denom_atoms = atom_mask.sum(1).clamp(min=1)[:, None, None]
        centered = (pos - (pos * atom_mask[..., None]).sum(1, keepdim=True) / denom_atoms) * atom_mask[..., None]
        has_atoms = atom_mask.any(dim=1)
        safe_mask = atom_mask.clone()
        safe_mask[~has_atoms, 0] = True
        F, P, pooled, logits = [], [], [], []
        for k in range(2):
            w = self.token_w[k](h).squeeze(-1).masked_fill(~text_mask, float("-inf")).softmax(-1)
            hk = torch.einsum("bt,btd->bd", w, h)
            lg = torch.einsum("bk,bnk->bn", self.q[k](hk), keys) / math.sqrt(self.key)
            lg = lg.masked_fill(~atom_mask, float("-inf"))
            a = torch.einsum("bk,bnk->bn", self.q[k](hk), keys) / math.sqrt(self.key)
            a = a.masked_fill(~safe_mask, float("-inf")).softmax(-1)
            if not self.training:
                a = torch.nn.functional.one_hot(a.argmax(-1), lg.shape[-1]).to(lg.dtype)
            a = a * has_atoms[:, None].to(a.dtype)
            F.append(torch.einsum("bn,bnc->bc", a, v))
            P.append(torch.einsum("bn,bnc->bc", a, centered))
            pooled.append(hk)
            logits.append(lg)
        cls = (h * text_mask[..., None]).sum(1) / text_mask.sum(1, keepdim=True).clamp(min=1)
        task_logits = self.task_head(torch.cat(pooled + [cls], -1))
        p = task_logits.softmax(-1)
        if not self.training:
            p = torch.nn.functional.one_hot(task_logits.argmax(-1), len(OPS)).to(p.dtype)
        u = P[1] - P[0]
        u = u / (u.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        y = p[:, 0:1] * F[0] + p[:, 1:2] * (F[0] - F[1]) + p[:, 2:3] * (F[0] * u).sum(-1, keepdim=True) * u
        return y, dict(task_logits=task_logits, ptr_logits=logits)

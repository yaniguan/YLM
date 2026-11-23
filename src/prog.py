"""ProgModel & GroupProgModel: learned semantic parser paired with an exact O(3) executor.

The text is mapped to a discrete program by learned invariant heads:
    task in {force, diff, proj} (or + {group} when allow_group=True)
    ptr1, ptr2   (two invariant-key atom pointer heads)
    group_mask   (optional invariant-key set-membership head for multi-atom functional groups)
and executed by exact O(3)-equivariant operators on a per-atom vector field V:
    force: V_1
    diff:  V_1 - V_2
    proj:  (V_1 . u) u,   u = unit(P_2 - P_1)
    group: sum_{i in G} m_i V_i   (functional-group / fragment net vector observable)

Training mixes operators with softmax task probabilities and soft pointers/masks;
eval mode switches to hard argmax task/pointer and thresholded set selection.
Output is O(3)-equivariant (rotations and reflections) when parity="o3".
"""
import math

import torch
import torch.nn as nn
from e3nn import o3

from src.encoder import Encoder
from src.irreps_to_text import invariants

OPS = ("force", "diff", "proj")
OPS_GROUP = ("force", "diff", "proj", "group")


class ProgModel(nn.Module):
    def __init__(
        self,
        text_dim,
        hidden_irreps="64x0e + 8x0o + 32x1o + 8x1e + 16x2e",
        n_layers=3,
        d=256,
        key=64,
        parity="o3",
        allow_group=False,
        encoder=None,
        text_dropout=0.0,
    ):
        super().__init__()
        if parity not in {"o3", "so3"}:
            raise ValueError("parity must be 'o3' or 'so3'")
        self.parity, self.key = parity, key
        self.text_dropout = float(text_dropout)
        self.allow_group = bool(allow_group)
        self.ops = OPS_GROUP if self.allow_group else OPS
        self.encoder = encoder if encoder is not None else Encoder(hidden_irreps=hidden_irreps, n_layers=n_layers)
        irreps = self.encoder.irreps_out
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d), nn.SiLU())
        self.token_w = nn.ModuleList([nn.Linear(d, 1) for _ in range(2)])
        self.q = nn.ModuleList([nn.Linear(d, key) for _ in range(2)])
        self.k = nn.Sequential(nn.Linear(irreps.num_irreps, 128), nn.SiLU(), nn.Linear(128, key))
        self.value = o3.Linear(irreps, "1x1o")
        n_streams = 4 if self.allow_group else 3
        if self.allow_group:
            self.group_w = nn.Linear(d, 1)
            self.group_q = nn.Linear(d, key)
            self.group_bias = nn.Parameter(torch.zeros(()))
        self.task_head = nn.Sequential(nn.Linear(n_streams * d, d), nn.SiLU(), nn.Linear(d, len(self.ops)))

    def forward(self, Z, pos, atom_mask, text, text_mask):
        t_in = text.to(pos.dtype)
        if self.text_dropout > 0.0 and self.training:
            t_in = nn.functional.dropout(t_in, p=self.text_dropout, training=True)
        h = self.text_proj(t_in)
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
            raw_lg = torch.einsum("bk,bnk->bn", self.q[k](hk), keys) / math.sqrt(self.key)
            lg = raw_lg.masked_fill(~atom_mask, float("-inf"))
            a = raw_lg.masked_fill(~safe_mask, float("-inf")).softmax(-1)
            if not self.training:
                a = torch.nn.functional.one_hot(a.argmax(-1), lg.shape[-1]).to(lg.dtype)
            a = a * has_atoms[:, None].to(a.dtype)
            F.append(torch.einsum("bn,bnc->bc", a, v))
            P.append(torch.einsum("bn,bnc->bc", a, centered))
            pooled.append(hk)
            logits.append(lg)

        cls = (h * text_mask[..., None]).sum(1) / text_mask.sum(1, keepdim=True).clamp(min=1)
        aux = dict(ptr_logits=logits)

        if self.allow_group:
            wg = self.group_w(h).squeeze(-1).masked_fill(~text_mask, float("-inf")).softmax(-1)
            hg = torch.einsum("bt,btd->bd", wg, h)
            pooled.append(hg)
            grp_raw = torch.einsum("bk,bnk->bn", self.group_q(hg), keys) / math.sqrt(self.key) + self.group_bias
            if self.training:
                gm = torch.sigmoid(grp_raw) * atom_mask.to(v.dtype)
            else:
                hard = (grp_raw > 0.0) & atom_mask
                empty_pred = has_atoms & (~hard.any(dim=-1))
                if empty_pred.any():
                    fallback = torch.nn.functional.one_hot(
                        grp_raw.masked_fill(~safe_mask, float("-inf")).argmax(-1),
                        grp_raw.shape[-1],
                    ).bool()
                    hard = torch.where(empty_pred[:, None], fallback & atom_mask, hard)
                gm = (hard & has_atoms[:, None]).to(v.dtype)
            F_grp = torch.einsum("bn,bnc->bc", gm, v)
            aux["group_logits"] = grp_raw
            aux["group_mask"] = gm

        task_logits = self.task_head(torch.cat(pooled + [cls], -1))
        aux["task_logits"] = task_logits
        p = task_logits.softmax(-1)
        if not self.training:
            p = torch.nn.functional.one_hot(task_logits.argmax(-1), len(self.ops)).to(p.dtype)
        u = P[1] - P[0]
        u = u / (u.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        y = p[:, 0:1] * F[0] + p[:, 1:2] * (F[0] - F[1]) + p[:, 2:3] * (F[0] * u).sum(-1, keepdim=True) * u
        if self.allow_group:
            y = y + p[:, 3:4] * F_grp
        return y, aux


class GroupProgModel(ProgModel):
    """ProgModel with multi-atom functional-group set readout (`allow_group=True`)."""

    def __init__(self, text_dim, **kwargs):
        kwargs.setdefault("allow_group", True)
        super().__init__(text_dim, **kwargs)

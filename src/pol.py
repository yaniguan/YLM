"""Polarizability field + exact tensor executor for TensorQA-pol.

PolModel predicts the static polarizability of a molecule as alpha0 (0e) + alpha2 (2e, traceless part), atom-additive:
    alpha0 = mu + sd * sum_i  o3.Linear(x_i) -> 0e
    alpha2 = s2 * ( sum_i o3.Linear(x_i) -> 2e  +  g * sum_i w_i(inv) |r_i|^2 Y_2(r_i) )     (r_i about the charge centre)
Question programs (task, atom A, atom B) are given by an oracle / rule parser, or predicted by learned heads
(task classifier + two atom pointers); the answer is then an EXACT contraction of the predicted tensor:
    pol_iso: alpha0     pol_aniso: alpha2     pol_dir: u^T alpha u     pol_ind: alpha u        (u = unit(r_B - r_A))
"""
import math

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.io import CartesianTensor

from src.encoder import Encoder
from src.irreps_to_text import invariants

CART = CartesianTensor("ij=ji")
OPS = ("pol_iso", "pol_aniso", "pol_dir", "pol_ind")
DIMS = {"pol_iso": 1, "pol_aniso": 5, "pol_dir": 1, "pol_ind": 3}


def to_matrix(alpha0, alpha2):
    """(B,), (B,5) -> (B,3,3): alpha = alpha0 I + dev(alpha2)"""
    full = torch.cat([torch.zeros_like(alpha0)[:, None], alpha2], -1)
    dev = CART.to_cartesian(full)
    return dev + alpha0[:, None, None] * torch.eye(3, dtype=dev.dtype, device=dev.device)


def execute_pol(alpha0, alpha2, pos, ptask, atoms):
    """Exact answers (B,5), zero padded; ptask codes index OPS; -1 or unresolved atoms give zeros."""
    b = torch.arange(pos.shape[0], device=pos.device)
    ia, ib = atoms[:, 0].clamp(min=0), atoms[:, 1].clamp(min=0)
    u = pos[b, ib] - pos[b, ia]
    u = u / (u.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    A = to_matrix(alpha0, alpha2)
    Au = torch.einsum("bij,bj->bi", A, u)
    pad = lambda v: torch.nn.functional.pad(v, (0, 5 - v.shape[-1]))        # noqa: E731
    out = torch.zeros(pos.shape[0], 5, dtype=alpha2.dtype, device=pos.device)
    out = torch.where((ptask == 0)[:, None], pad(alpha0[:, None]), out)
    out = torch.where((ptask == 1)[:, None], alpha2, out)
    out = torch.where((ptask == 2)[:, None], pad((u * Au).sum(-1, keepdim=True)), out)
    out = torch.where((ptask == 3)[:, None], pad(Au), out)
    needs_atoms = (ptask >= 2)
    ok = (ptask >= 0) & (~needs_atoms | ((atoms >= 0).all(-1)))
    return out * ok[:, None]


class PolModel(nn.Module):
    def __init__(self, text_dim, hidden_irreps="64x0e + 8x0o + 32x1o + 8x1e + 16x2e", n_layers=3, d=256, key=64,
                 parity="o3", iso_mean=0.0, iso_std=1.0, s2=1.0, mean_atoms=1.0):
        super().__init__()
        self.parity, self.key = parity, key
        self.register_buffer("iso", torch.tensor([iso_mean, iso_std, s2, mean_atoms], dtype=torch.float32))
        self.encoder = Encoder(hidden_irreps=hidden_irreps, n_layers=n_layers, n_elements=18)
        irreps = self.encoder.irreps_out
        self.head0 = o3.Linear(irreps, "1x0e")
        self.head2 = o3.Linear(irreps, "1x2e")
        self.w = nn.Sequential(nn.Linear(irreps.num_irreps, 128), nn.SiLU(), nn.Linear(128, 1))
        self.g = nn.Parameter(torch.tensor(0.1))
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d), nn.SiLU())
        self.token_w = nn.ModuleList([nn.Linear(d, 1) for _ in range(2)])
        self.q = nn.ModuleList([nn.Linear(d, key) for _ in range(2)])
        self.k = nn.Sequential(nn.Linear(irreps.num_irreps, 128), nn.SiLU(), nn.Linear(128, key))
        self.task_head = nn.Sequential(nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, len(OPS)))

    def tensor(self, Z, pos, mask):
        m = mask[..., None].to(pos.dtype)
        x = self.encoder(Z, pos, mask)
        inv = invariants(x, self.encoder.irreps_out, self.parity)
        mu, sd, s2, mean_atoms = self.iso.tolist()
        a0 = mu * mask.sum(1).to(pos.dtype) / mean_atoms + sd * (self.head0(x) * m).sum(1).squeeze(-1)
        Zf = Z.to(pos.dtype) * mask
        centre = (Zf[..., None] * pos).sum(1, keepdim=True) / Zf.sum(1)[:, None, None]
        r = (pos - centre) * m
        Y = o3.spherical_harmonics(2, r, normalize=False, normalization="integral")
        a2 = s2 * ((self.head2(x) * m).sum(1) + self.g * (self.w(inv) * Y * m).sum(1))
        return a0, a2, inv

    def program(self, text, tmask, inv, mask):
        h = self.text_proj(text)
        keys = self.k(inv)
        pooled, logits = [], []
        for k in range(2):
            w = self.token_w[k](h).squeeze(-1).masked_fill(~tmask, float("-inf")).softmax(-1)
            hk = torch.einsum("bt,btd->bd", w, h)
            lg = (torch.einsum("bk,bnk->bn", self.q[k](hk), keys) / math.sqrt(self.key)).masked_fill(~mask, float("-inf"))
            pooled.append(hk)
            logits.append(lg)
        cls = (h * tmask[..., None]).sum(1) / tmask.sum(1, keepdim=True)
        return self.task_head(torch.cat(pooled + [cls], -1)), logits

    def forward(self, Z, pos, mask, text, tmask):
        a0, a2, inv = self.tensor(Z, pos, mask)
        task_logits, ptr = self.program(text.to(pos.dtype), tmask, inv, mask)
        return a0, a2, task_logits, ptr

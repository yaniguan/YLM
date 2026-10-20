"""l=2 output: the traceless molecular quadrupole as five 2e coefficients.

Q (3x3, symmetric, traceless) rotates as R Q R^T. e3nn's CartesianTensor("ij=ji") maps it to 0e+2e; we keep the
2e part (the 0e trace is zero by construction), which rotates with the Wigner-D matrix of "2e".

QuadModel: encoder -> sum over atoms of
    o3.Linear(x) -> 2e                           (learned equivariant l=2 feature of the atom)
  + w_i(invariants, text) * |r_i|^2 Y_2(r_i)      (geometric l=2 term, r_i relative to the charge centre)
The text gates the two terms with invariant scalars (text-conditioned, still exactly equivariant).
"""
import math

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.io import CartesianTensor

from src.encoder import Encoder
from src.irreps_to_text import invariants

CART = CartesianTensor("ij=ji")


def quad_to_2e(Q):
    """(..., 3, 3) symmetric -> (..., 5) coefficients of 2e (drops the trace)"""
    return CART.from_cartesian(Q)[..., 1:]


class QuadModel(nn.Module):
    def __init__(self, text_dim, hidden_irreps="64x0e + 8x0o + 32x1o + 8x1e + 16x2e", n_layers=3, d=128):
        super().__init__()
        self.encoder = Encoder(hidden_irreps=hidden_irreps, n_layers=n_layers)
        irreps = self.encoder.irreps_out
        self.lin = o3.Linear(irreps, "1x2e")
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d), nn.SiLU())
        self.w = nn.Sequential(nn.Linear(irreps.num_irreps + d, 128), nn.SiLU(), nn.Linear(128, 1))
        self.gate = nn.Linear(d, 2)

    def forward(self, Z, pos, atom_mask, text, text_mask):
        m = atom_mask[..., None].to(pos.dtype)
        h = self.text_proj(text.to(pos.dtype))
        t = (h * text_mask[..., None]).sum(1) / text_mask.sum(1, keepdim=True)          # (B, d) invariant
        x = self.encoder(Z, pos, atom_mask)
        g = self.gate(t)                                                                 # (B, 2) invariant
        learned = (self.lin(x) * m).sum(1)                                               # (B, 5)
        Zf = Z.to(pos.dtype) * atom_mask
        centre = (Zf[..., None] * pos).sum(1, keepdim=True) / Zf.sum(1)[:, None, None]
        r = (pos - centre) * m
        norm2 = r.pow(2).sum(-1, keepdim=True)
        Y = o3.spherical_harmonics(2, r, normalize=False, normalization="integral")      # |r|^2 Y_2(r_hat), (B,N,5)
        w = self.w(torch.cat([invariants(x, self.encoder.irreps_out), t[:, None].expand(-1, x.shape[1], -1)], -1))
        geom = (w * Y * m).sum(1)
        return g[:, :1] * learned + g[:, 1:] * geom

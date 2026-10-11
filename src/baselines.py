"""Two controlled baselines for TensorQA-v0 (see 04_data_and_benchmark.md).

PipelineForce  "text -> atom index -> force model"
    A rule-based parser turns the question into an atom index (src/parse_question.py).
    The model never sees the text. It reads the per-atom equivariant features at that atom
    (force) or sums them over atoms (dipole). Equivariant, so the only thing language does is
    the parser. Answers: can the same task be solved without a language model at all?

ScalarCollapse  "invariants only into the language side"
    Same encoder, but only invariants (scalars + lengths of l>=1 channels) leave it, and a
    plain MLP writes 3 numbers in the lab frame. This is a proxy for the invariant-token route
    (EquiLLM-style adapters, Mol-StrucTok-style spherical tokens), NOT a reproduction of them.
    An invariant input has no direction, so the output cannot follow the molecule's rotation;
    the experiment shows how large that gap is.
"""
import math

import torch
import torch.nn as nn
from e3nn import o3

from src.encoder import Encoder
from src.irreps_to_text import invariants


class PipelineForce(nn.Module):
    def __init__(self, hidden_irreps="64x0e + 8x0o + 32x1o + 8x1e + 16x2e", n_layers=3):
        super().__init__()
        self.encoder = Encoder(hidden_irreps=hidden_irreps, n_layers=n_layers)
        self.readout = o3.Linear(self.encoder.irreps_out, "1x1o")

    def forward(self, Z, pos, atom_mask, text, text_mask, task, atom):
        v = self.readout(self.encoder(Z, pos, atom_mask)) * atom_mask[..., None]   # (B, N, 3)
        dipole = v.sum(1)
        idx = atom.clamp(min=0)
        force = v[torch.arange(v.shape[0], device=v.device), idx]
        onehot = torch.zeros(v.shape[:2], device=v.device)
        onehot[torch.arange(v.shape[0], device=v.device), idx] = 1.0
        return torch.where((task == 1)[:, None], force, dipole), onehot


class ScalarCollapse(nn.Module):
    def __init__(self, text_dim, hidden_irreps="64x0e + 8x0o + 32x1o + 8x1e + 16x2e", n_layers=3, d=128):
        super().__init__()
        self.encoder = Encoder(hidden_irreps=hidden_irreps, n_layers=n_layers)
        n_inv = self.encoder.irreps_out.num_irreps
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d), nn.SiLU())
        self.atom_proj = nn.Sequential(nn.Linear(n_inv, d), nn.SiLU(), nn.Linear(d, 2 * d))
        self.token_weight = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, 3))
        self.d = d

    def forward(self, Z, pos, atom_mask, text, text_mask, task, atom=None):
        h = self.text_proj(text)                                                    # (B, T, d)
        x = self.encoder(Z, pos, atom_mask)
        k, v = self.atom_proj(invariants(x, self.encoder.irreps_out)).split(self.d, dim=-1)
        scores = torch.einsum("btd,bnd->btn", h, k) / math.sqrt(self.d)
        attn = scores.masked_fill(~atom_mask[:, None, :], float("-inf")).softmax(-1)
        picked = torch.einsum("btn,bnd->btd", attn, v)
        w = self.token_weight(h).squeeze(-1).masked_fill(~text_mask, float("-inf")).softmax(-1)
        return self.head(torch.cat([torch.einsum("bt,btd->bd", w, picked), h.mean(1)], -1)), \
            torch.einsum("bt,btn->bn", w, attn)

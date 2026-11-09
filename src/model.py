"""YLM with configurable A/B/C cross-attention.

  question tokens (frozen text encoder, precomputed) ─┐
                                                      ├─► IrrepsToText ─► e_t per token
  atoms ─► Encoder (equivariant) ─► irreps per atom ──┘
                                                          │
            pool tokens with invariant weights  ◄─────────┘
                                                          │
            default: o3.Linear -> 1x1o  (one vector: dipole or force)
            tensor_readout=True: 0e+2e -> one symmetric 3x3 matrix

option: A invariant keys; B direction-aware keys; C text-weighted CG product.
legacy_odd_scalars: reproduce raw 0o keys for parity ablations (SO(3) only).
tensor_readout: off by default, so v0 checkpoints still load.

text_mode:
  "text"   : use the precomputed token states of the question (MiniLM / MatSciBERT)
  "taskid" : baseline v. Throw the question away, use a learned token for the task
             ("dipole" or "force"). The model then cannot know WHICH atom is asked about.
"""
import torch
import torch.nn as nn
from e3nn import o3

from src.encoder import Encoder
from src.irreps_to_text import IrrepsToText
from src.tensor_readout import TensorReadout


def _has_irrep(irreps, l, p):
    return any(ir.l == l and ir.p == p for _, ir in o3.Irreps(irreps))


class YLM(nn.Module):
    def __init__(self, text_dim, text_mode="text", hidden_irreps="64x0e + 8x0o + 32x1o + 8x1e + 16x2e",
                 cross_irreps="32x0e + 32x1o + 8x1e + 8x2e", n_layers=3, n_tasks=2, option="A",
                 legacy_odd_scalars=False, tensor_readout=False):
        super().__init__()
        self.text_mode = text_mode
        self.tensor_readout = bool(tensor_readout)
        self.encoder = Encoder(hidden_irreps=hidden_irreps, n_layers=n_layers)
        self.task_embed = nn.Embedding(n_tasks, text_dim)             # only used in "taskid"
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, 256), nn.SiLU())
        self.cross = IrrepsToText(256, self.encoder.irreps_out, cross_irreps, option=option,
                                  legacy_odd_scalars=legacy_odd_scalars)
        self.token_weight = nn.Linear(256, 1)
        if self.tensor_readout:
            if not (_has_irrep(cross_irreps, 0, 1) and _has_irrep(cross_irreps, 2, 1)):
                raise ValueError("tensor readout needs 0e and 2e in cross_irreps")
            self.readout = TensorReadout(cross_irreps)
        else:
            self.readout = o3.Linear(cross_irreps, "1x1o")

    def forward(self, Z, pos, atom_mask, text, text_mask, task):
        if self.text_mode == "taskid":
            text = self.task_embed(task)[:, None, :]                  # (B, 1, d)
            text_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
        h = self.text_proj(text)                                      # (B, T, 256) invariant
        x = self.encoder(Z, pos, atom_mask)                           # (B, N, D)   equivariant
        e, attn = self.cross(h, x, atom_mask)                         # (B, T, D')  equivariant

        w = self.token_weight(h).squeeze(-1).masked_fill(~text_mask, float("-inf")).softmax(-1)
        e_pool = torch.einsum("bt,btd->bd", w, e)                     # (B, D')
        attn_pool = torch.einsum("bt,btn->bn", w, attn)               # (B, N)  for analysis
        # Default answer is (B, 3). tensor_readout=True returns (B, 3, 3).
        return self.readout(e_pool), attn_pool

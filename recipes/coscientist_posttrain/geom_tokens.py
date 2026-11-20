"""Geometry -> k soft tokens of LLM width.

(Z, pos in bohr) -> YLM Encoder (per-atom irreps) -> IrrepsToText with k learned queries
(the queries play the role of text states, so the module works as an attention pooler)
-> invariants -> MLP projector -> (G, k, d_llm).

Only invariants enter the LLM, so the soft tokens do not change under rotation, reflection,
translation or atom permutation. The equivariant pooled features are returned too, for an
optional vector readout.
"""
import torch
import torch.nn as nn

from src.encoder import Encoder
from src.irreps_to_text import IrrepsToText, invariants


class GeomSoftTokens(nn.Module):
    def __init__(self, d_llm, k=8, hidden_irreps="16x0e + 4x0o + 8x1o + 4x1e + 4x2e",
                 n_layers=2, query_dim=32, n_elements=100):
        super().__init__()
        self.k = k
        self.encoder = Encoder(hidden_irreps, n_layers=n_layers, n_elements=n_elements)
        self.queries = nn.Parameter(torch.randn(k, query_dim) * 0.5)
        self.pool = IrrepsToText(query_dim, self.encoder.irreps_out, self.encoder.irreps_out)
        n_inv = self.pool.out_irreps.num_irreps
        self.proj = nn.Sequential(nn.Linear(n_inv, d_llm), nn.SiLU(), nn.Linear(d_llm, d_llm))

    def forward(self, geom):
        """geom: dict from src.ylm_lm.collate_geometry. Returns (soft (G, k, d_llm), e (G, k, irreps))."""
        mask = geom["atom_mask"]
        x = self.encoder(geom["Z"], geom["pos"], mask)
        h = self.queries.expand(mask.shape[0], -1, -1)
        e, _ = self.pool(h, x, mask)
        return self.proj(invariants(e, self.pool.out_irreps)), e


def scatter_into(embeds, slots, soft):
    """Write soft tokens (G, k, d) into embeds (1, T, d) at flat positions slots (G, k)."""
    if slots.numel() == 0:
        return embeds
    out = embeds.clone()
    out[0, slots.reshape(-1)] = soft.reshape(-1, soft.shape[-1]).to(out.dtype)
    return out

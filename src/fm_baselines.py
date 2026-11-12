"""Baseline 3D models compared side by side against `YLMUniverseModel`:

1. `UniMolStyle3DBaseline` (Zhou et al., ICLR 2023):
   3D Transformer built from atom-to-atom distances `d_ij = ||r_i - r_j||`. Because distances do not
   change under rotation or mirror reflection, its scalar features cannot rotate arrows or tell `(R)` from `(S)`.

2. `GraphMVP3DInfoMaxBaseline` (Liu et al., ICLR 2022; Stark et al., ICML 2022):
   2D bond-graph network trained to match a 3D teacher during pretraining, using only 2D bonds at test time.

3. `EquiLLMMultiTaskBaseline`:
   Passes only plain numbers through the text bridge and scales atom 3D vectors at the end.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

from src.encoder import Encoder
from src.irreps_to_text import invariants
from src.toolkit.ingest import BOHR_TO_ANGSTROM, COVALENT_RADII_ANG
from src.universe_fm import LOT_VOCAB, _INV_SOFTPLUS_ONE


def _pairwise_mic_torch(
    pos: torch.Tensor,
    atom_mask: torch.Tensor,
    cell: Optional[torch.Tensor] = None,
    pbc: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute batched pairwise r_i - r_j (B, N, N, 3) and distances (B, N, N) in bohr with optional PBC MIC."""
    B, N, _ = pos.shape
    rel = pos[:, :, None, :] - pos[:, None, :, :]
    if cell is not None:
        if cell.ndim == 2:
            cell = cell.unsqueeze(0).expand(B, 3, 3)
        pbc_mask = (cell.norm(dim=-1) > 0) if pbc is None else pbc.to(dtype=torch.bool, device=pos.device)
        if pbc_mask.ndim == 1:
            pbc_mask = pbc_mask.unsqueeze(0).expand(B, 3)
        wrapped = []
        for b in range(B):
            pb = pbc_mask[b]
            rb = rel[b]
            if pb.any():
                basis = cell[b, pb]
                if (torch.linalg.svdvals(basis) > 1e-7).all():
                    pinv = torch.linalg.pinv(basis)
                    frac = (rb.reshape(-1, 3) @ pinv).detach()
                    rb = (rb.reshape(-1, 3) - torch.round(frac) @ basis).reshape(N, N, 3)
            wrapped.append(rb)
        rel = torch.stack(wrapped, dim=0)
    pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]
    rel = rel * pair_mask[..., None].to(pos.dtype)
    eye = torch.eye(N, dtype=torch.bool, device=pos.device)[None, :, :]
    dist = (rel.pow(2).sum(dim=-1) + 1e-12).sqrt() * (pair_mask & ~eye).to(pos.dtype)
    return rel, dist


class _GaussianDistancePairEncoder(nn.Module):
    """Uni-Mol Gaussian basis expansion of pairwise distances d_ij with pair-species affine modulation."""

    def __init__(self, n_elements: int = 100, n_kernels: int = 32, cutoff: float = 12.0):
        super().__init__()
        self.n_kernels = n_kernels
        self.pair_scale = nn.Embedding(n_elements, 8)
        self.pair_shift = nn.Embedding(n_elements, 8)
        nn.init.zeros_(self.pair_scale.weight)
        nn.init.zeros_(self.pair_shift.weight)
        self.edge_proj = nn.Linear(8, 2)
        nn.init.zeros_(self.edge_proj.weight)
        nn.init.zeros_(self.edge_proj.bias)

        centers = torch.linspace(0.5, cutoff, n_kernels)
        widths = torch.full((n_kernels,), (cutoff / n_kernels))
        self.means = nn.Parameter(centers)
        self.log_stds = nn.Parameter(widths.clamp(min=0.15).log())

    def forward(self, Z: torch.Tensor, dist: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        z_sum = self.pair_scale(Z)[:, :, None, :] + self.pair_scale(Z)[:, None, :, :]
        aff = self.edge_proj(torch.tanh(z_sum))
        gamma = F.softplus(aff[..., 0] + _INV_SOFTPLUS_ONE)
        beta = aff[..., 1]
        d_mod = gamma * dist + beta
        stds = self.log_stds.exp().clamp(min=0.08)
        rbf = torch.exp(-0.5 * ((d_mod[..., None] - self.means) / stds).pow(2))
        return rbf * pair_mask[..., None].to(dist.dtype)


class _UniMol3DLayer(nn.Module):
    """One SE(3)-invariant 3D Transformer block with persistent pair-representation attention bias."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        pair_repr: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        m = atom_mask[..., None].to(x.dtype)
        qkv = (self.qkv(self.norm1(x)) * m).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, H, N, d_head)

        attn_logits = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.d_head)
        pair_next = pair_repr + attn_logits

        has_atom = atom_mask.any(dim=-1, keepdim=True)
        safe_mask = atom_mask.clone()
        safe_mask[~has_atom.squeeze(-1), 0] = True
        scores = pair_next.masked_fill(~safe_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v).transpose(1, 2).reshape(B, N, D)
        x = (x + self.out_proj(out)) * m
        x = (x + self.ffn(self.norm2(x))) * m
        return x, pair_next


class _InvariantLanguageHead(nn.Module):
    """Shared language->atom cross-attention readout mapping invariant atom features to scalar/vector/tensor."""

    def __init__(self, text_dim: int, d_model: int, n_scalars: int = 1, n_lots: int = len(LOT_VOCAB)):
        super().__init__()
        self.d_model = d_model
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d_model), nn.SiLU())
        self.lot_embed = nn.Embedding(n_lots, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.gate_pick = nn.Linear(d_model, d_model)
        self.gate_sum = nn.Linear(d_model, d_model)
        self.token_weight = nn.Linear(d_model, 1)

        self.scalar_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_scalars),
        )
        self.vector_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),
        )
        self.tensor_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 9),
        )
        self.chiral_head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        atom_feats: torch.Tensor,
        atom_mask: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        lot_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h = (self.text_proj(text) + self.lot_embed(lot_id)[:, None, :]) * text_mask[..., None].to(atom_feats.dtype)
        q = self.q(h)
        k = self.k(atom_feats)
        v = self.v(atom_feats) * atom_mask[..., None].to(atom_feats.dtype)

        has_atom = atom_mask.any(dim=-1, keepdim=True)
        safe_amask = atom_mask.clone()
        safe_amask[~has_atom.squeeze(-1), 0] = True
        scores = (torch.einsum("btd,bnd->btn", q, k) / math.sqrt(self.d_model)).masked_fill(
            ~safe_amask[:, None, :], float("-inf")
        )
        attn = scores.softmax(dim=-1)

        picked = torch.einsum("btn,bnd->btd", attn, v)
        count = atom_mask.sum(dim=1, keepdim=True).clamp(min=1).to(atom_feats.dtype)
        mean_v = (v.sum(dim=1) / count)[:, None, :]
        tok_fused = torch.sigmoid(self.gate_pick(h)) * picked + self.gate_sum(h) * mean_v

        has_tok = text_mask.any(dim=-1, keepdim=True)
        safe_tmask = text_mask.clone()
        safe_tmask[~has_tok.squeeze(-1), 0] = True
        w = self.token_weight(h).squeeze(-1).masked_fill(~safe_tmask, float("-inf")).softmax(dim=-1)

        pooled_geom = torch.einsum("bt,btd->bd", w, tok_fused)
        pooled_text = torch.einsum("bt,btd->bd", w, h)
        attn_pool = torch.einsum("bt,btn->bn", w, attn)

        joint = torch.cat([pooled_geom, pooled_text], dim=-1)
        scalar_out = self.scalar_head(joint)
        chiral_out = self.chiral_head(joint)
        vector_out = self.vector_head(joint)
        raw_m = self.tensor_head(joint).reshape(-1, 3, 3)
        tensor_out = 0.5 * (raw_m + raw_m.transpose(-1, -2))
        return {
            "scalar": scalar_out,
            "chiral": chiral_out,
            "vector": vector_out,
            "tensor": tensor_out,
            "attn": attn_pool,
        }


class UniMolStyle3DBaseline(nn.Module):
    """Uni-Mol style SE(3)-invariant distance-matrix 3D Transformer + coordinate denoising + language readout."""

    def __init__(
        self,
        text_dim: int = 128,
        *,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        n_kernels: int = 32,
        n_elements: int = 100,
        n_scalars: int = 1,
        cutoff: float = 12.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.atom_embed = nn.Embedding(n_elements, d_model)
        self.lot_atom = nn.Embedding(len(LOT_VOCAB), d_model)
        self.dist_encoder = _GaussianDistancePairEncoder(n_elements=n_elements, n_kernels=n_kernels, cutoff=cutoff)
        self.pair_proj = nn.Sequential(nn.Linear(n_kernels, n_kernels), nn.GELU(), nn.Linear(n_kernels, n_heads))
        self.layers = nn.ModuleList([_UniMol3DLayer(d_model, n_heads) for _ in range(n_layers)])
        self.coord_pair_head = nn.Sequential(nn.Linear(n_heads, n_heads), nn.GELU(), nn.Linear(n_heads, 1))
        self.readout = _InvariantLanguageHead(text_dim, d_model, n_scalars=n_scalars)

    def encode_3d(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        atom_mask: torch.Tensor,
        lot_id: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N = Z.shape
        if lot_id is None:
            lot_id = torch.zeros(B, dtype=torch.long, device=Z.device)
        rel, dist = _pairwise_mic_torch(pos, atom_mask, cell=cell, pbc=pbc)
        pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :] & ~torch.eye(N, dtype=torch.bool, device=pos.device)[None]
        rbf = self.dist_encoder(Z, dist, pair_mask)
        pair_repr = self.pair_proj(rbf).permute(0, 3, 1, 2)  # (B, H, N, N)

        x = (self.atom_embed(Z) + self.lot_atom(lot_id)[:, None, :]) * atom_mask[..., None].to(pos.dtype)
        for layer in self.layers:
            x, pair_repr = layer(x, pair_repr, atom_mask)
        return x, pair_repr, rel, dist

    def predict_denoise(
        self,
        Z: torch.Tensor,
        pos_noisy: torch.Tensor,
        atom_mask: torch.Tensor,
        lot_id: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, pair_repr, rel, dist = self.encode_3d(Z, pos_noisy, atom_mask, lot_id=lot_id, cell=cell, pbc=pbc)
        N = Z.shape[1]
        pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :] & ~torch.eye(N, dtype=torch.bool, device=pos_noisy.device)[None]
        w_ij = self.coord_pair_head(pair_repr.permute(0, 2, 3, 1)).squeeze(-1) * pair_mask.to(pos_noisy.dtype)
        unit_rel = rel / dist.clamp(min=1e-4)[..., None]
        counts = pair_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(pos_noisy.dtype)
        delta = (w_ij[..., None] * unit_rel).sum(dim=2) / counts
        return delta * atom_mask[..., None].to(pos_noisy.dtype)

    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        atom_mask: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        lot_id: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if lot_id is None:
            lot_id = torch.zeros(Z.shape[0], dtype=torch.long, device=Z.device)
        x, pair_repr, rel, dist = self.encode_3d(Z, pos, atom_mask, lot_id=lot_id, cell=cell, pbc=pbc)
        out = self.readout(x, atom_mask, text, text_mask, lot_id)

        N = Z.shape[1]
        pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :] & ~torch.eye(N, dtype=torch.bool, device=pos.device)[None]
        w_ij = self.coord_pair_head(pair_repr.permute(0, 2, 3, 1)).squeeze(-1) * pair_mask.to(pos.dtype)
        unit_rel = rel / dist.clamp(min=1e-4)[..., None]
        counts = pair_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(pos.dtype)
        out["denoise"] = ((w_ij[..., None] * unit_rel).sum(dim=2) / counts) * atom_mask[..., None].to(pos.dtype)
        return out


def info_nce_3d_2d_loss(z_2d: torch.Tensor, z_3d: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric InfoNCE mutual-information contrastive loss aligning 2D graph and 3D geometric embeddings."""
    if z_2d.shape[0] <= 1:
        return (1.0 - F.cosine_similarity(z_2d, z_3d, dim=-1)).mean()
    u = F.normalize(z_2d, dim=-1, eps=1e-8)
    v = F.normalize(z_3d, dim=-1, eps=1e-8)
    logits = (u @ v.T) / max(temperature, 1e-4)
    labels = torch.arange(u.shape[0], device=u.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


class _TopologicalGINLayer(nn.Module):
    """2D Graph Isomorphism / MPNN layer operating strictly on binary connectivity A_ij (no 3D distances/angles)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, h: torch.Tensor, adj: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        agg = torch.einsum("bij,bjd->bid", adj, h) / deg.sqrt()
        out = h + self.mlp((1.0 + self.eps) * h + agg)
        return out * atom_mask[..., None].to(h.dtype)


class GraphMVP3DInfoMaxBaseline(nn.Module):
    """GraphMVP / 3D-InfoMax baseline: 2D topological GNN distilled from a 3D teacher via InfoNCE."""

    def __init__(
        self,
        text_dim: int = 128,
        *,
        d_model: int = 128,
        n_layers: int = 3,
        n_elements: int = 100,
        n_scalars: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.atom_embed = nn.Embedding(n_elements, d_model)
        self.lot_embed = nn.Embedding(len(LOT_VOCAB), d_model)
        self.gin_layers = nn.ModuleList([_TopologicalGINLayer(d_model) for _ in range(n_layers)])
        self.proj_2d = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

        # 3D Teacher (used ONLY for contrastive InfoNCE loss during training, discarded at 2D inference!)
        self.teacher_3d = UniMolStyle3DBaseline(
            text_dim=text_dim, d_model=d_model, n_layers=2, n_heads=4, n_elements=n_elements, n_scalars=n_scalars
        )
        self.proj_3d = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.readout = _InvariantLanguageHead(text_dim, d_model, n_scalars=n_scalars)

        radii = torch.full((max(n_elements, 119),), 1.40, dtype=torch.float32)
        for z_val, r_ang in COVALENT_RADII_ANG.items():
            if z_val < radii.numel():
                radii[z_val] = float(r_ang)
        self.register_buffer("_cov_radii_bohr", radii / BOHR_TO_ANGSTROM, persistent=False)

    def _build_2d_adjacency(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        atom_mask: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract binary covalent/coordination adjacency A_ij in {0, 1} (discards continuous 3D geometry)."""
        with torch.no_grad():
            _, dist = _pairwise_mic_torch(pos, atom_mask, cell=cell, pbc=pbc)
            zc = Z.clamp(min=0, max=self._cov_radii_bohr.numel() - 1)
            r_sum = self._cov_radii_bohr[zc][:, :, None] + self._cov_radii_bohr[zc][:, None, :]
            # Allow alkali/transition-metal coordination bonds up to ~2.55 A (4.82 bohr)
            is_metal = (zc == 3) | (zc == 11) | (zc == 12) | ((zc >= 22) & (zc <= 30))
            coord_pair = is_metal[:, :, None] | is_metal[:, None, :]
            thresh = torch.where(coord_pair, torch.clamp(1.38 * r_sum, min=4.82), 1.38 * r_sum)
            N = Z.shape[1]
            valid = atom_mask[:, :, None] & atom_mask[:, None, :] & ~torch.eye(N, dtype=torch.bool, device=pos.device)[None]
            adj = ((dist > 1e-6) & (dist <= thresh) & valid).to(pos.dtype)
        return adj

    def encode_2d(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        atom_mask: torch.Tensor,
        lot_id: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adj = self._build_2d_adjacency(Z, pos, atom_mask, cell=cell, pbc=pbc)
        h = (self.atom_embed(Z) + self.lot_embed(lot_id)[:, None, :]) * atom_mask[..., None].to(pos.dtype)
        for layer in self.gin_layers:
            h = layer(h, adj, atom_mask)
        counts = atom_mask.sum(dim=1, keepdim=True).clamp(min=1).to(pos.dtype)
        g_2d = self.proj_2d((h * atom_mask[..., None].to(pos.dtype)).sum(dim=1) / counts)
        return h, g_2d

    def contrastive_3d_2d_loss(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        atom_mask: torch.Tensor,
        lot_id: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        if lot_id is None:
            lot_id = torch.zeros(Z.shape[0], dtype=torch.long, device=Z.device)
        _, z_2d = self.encode_2d(Z, pos, atom_mask, lot_id, cell=cell, pbc=pbc)
        x_3d, _, _, _ = self.teacher_3d.encode_3d(Z, pos, atom_mask, lot_id=lot_id, cell=cell, pbc=pbc)
        counts = atom_mask.sum(dim=1, keepdim=True).clamp(min=1).to(pos.dtype)
        z_3d = self.proj_3d((x_3d * atom_mask[..., None].to(pos.dtype)).sum(dim=1) / counts)
        return info_nce_3d_2d_loss(z_2d, z_3d, temperature=temperature)

    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        atom_mask: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        lot_id: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if lot_id is None:
            lot_id = torch.zeros(Z.shape[0], dtype=torch.long, device=Z.device)
        h_2d, z_2d = self.encode_2d(Z, pos, atom_mask, lot_id, cell=cell, pbc=pbc)
        out = self.readout(h_2d, atom_mask, text, text_mask, lot_id)
        out["z_2d"] = z_2d
        out["denoise"] = torch.zeros(Z.shape[0], Z.shape[1], 3, dtype=pos.dtype, device=pos.device)
        return out


class EquiLLMMultiTaskBaseline(nn.Module):
    """Multi-task EquiLLM control: invariant Transformer bridge + post-hoc 1o vector bypass (no IrrepsToText CG)."""

    def __init__(
        self,
        text_dim: int = 128,
        *,
        hidden_irreps: str = "64x0e + 8x0o + 32x1o + 8x1e + 16x2e",
        n_layers: int = 3,
        d_model: int = 128,
        n_vec: int = 32,
        n_bridge_layers: int = 2,
        n_heads: int = 4,
        n_elements: int = 100,
        n_scalars: int = 1,
    ):
        super().__init__()
        self.n_vec = n_vec
        self.d_model = d_model
        self.encoder = Encoder(hidden_irreps=hidden_irreps, n_layers=n_layers, n_elements=n_elements)
        n_inv = self.encoder.irreps_out.num_irreps

        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d_model), nn.SiLU())
        self.atom_proj = nn.Sequential(nn.LayerNorm(n_inv), nn.Linear(n_inv, d_model), nn.SiLU())
        self.lot_embed = nn.Embedding(len(LOT_VOCAB), d_model)
        self.type_embed = nn.Embedding(2, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.bridge = nn.TransformerEncoder(enc_layer, num_layers=n_bridge_layers, enable_nested_tensor=False)

        self.token_weight = nn.Linear(d_model, 1)
        self.atom_score = nn.Linear(d_model, 1)
        self.atom_gate = nn.Linear(d_model, n_vec)
        self.gate_pick = nn.Linear(d_model, n_vec)
        self.gate_sum = nn.Linear(d_model, n_vec)

        self.vec_bypass = o3.Linear(self.encoder.irreps_out, f"{n_vec}x1o")
        self.vec_readout = o3.Linear(f"{n_vec}x1o", "1x1o")
        self.ten_left = o3.Linear(f"{n_vec}x1o", f"{n_vec}x1o")
        self.ten_right = o3.Linear(f"{n_vec}x1o", f"{n_vec}x1o")
        self.ten_iso = nn.Linear(d_model, 1)
        self.scalar_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_scalars),
        )
        self.chiral_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.denoise_head = o3.Linear(self.encoder.irreps_out, "1x1o")

    def predict_denoise(
        self,
        Z: torch.Tensor,
        pos_noisy: torch.Tensor,
        atom_mask: torch.Tensor,
        lot_id: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.encoder(Z, pos_noisy, atom_mask, cell=cell, pbc=pbc)
        return self.denoise_head(x) * atom_mask[..., None].to(pos_noisy.dtype)

    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        atom_mask: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        lot_id: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        B, T, _ = text.shape
        N = Z.shape[1]
        if lot_id is None:
            lot_id = torch.zeros(B, dtype=torch.long, device=Z.device)

        x = self.encoder(Z, pos, atom_mask, cell=cell, pbc=pbc)
        lot_v = self.lot_embed(lot_id)[:, None, :]
        h_text = (self.text_proj(text) + lot_v + self.type_embed.weight[0]) * text_mask[..., None].to(pos.dtype)
        h_atom = (self.atom_proj(invariants(x, self.encoder.irreps_out)) + lot_v + self.type_embed.weight[1]) * atom_mask[..., None].to(pos.dtype)

        seq = torch.cat([h_text, h_atom], dim=1)
        valid = torch.cat([text_mask, atom_mask], dim=1)
        out = self.bridge(seq, src_key_padding_mask=~valid)
        z_text, z_atom = out[:, :T], out[:, T:]

        tw = self.token_weight(z_text).squeeze(-1).masked_fill(~text_mask, float("-inf")).softmax(dim=-1)
        z_global = torch.einsum("bt,btd->bd", tw, z_text)

        attn = self.atom_score(z_atom).squeeze(-1).masked_fill(~atom_mask, float("-inf")).softmax(dim=-1)
        a_gate = self.atom_gate(z_atom)
        v = (self.vec_bypass(x) * atom_mask[..., None].to(pos.dtype)).reshape(B, N, self.n_vec, 3)

        picked = torch.einsum("bn,bnc,bncm->bcm", attn, a_gate, v)
        total = v.sum(dim=1)
        fused = (torch.sigmoid(self.gate_pick(z_global))[..., None] * picked + self.gate_sum(z_global)[..., None] * total)
        fused_flat = fused.reshape(B, self.n_vec * 3)

        vec_out = self.vec_readout(fused_flat)
        vl = self.ten_left(fused_flat).reshape(B, self.n_vec, 3)
        vr = self.ten_right(fused_flat).reshape(B, self.n_vec, 3)
        outer = torch.einsum("bci,bcj->bij", vl, vr) / math.sqrt(self.n_vec)
        sym_outer = 0.5 * (outer + outer.transpose(-1, -2))
        iso = self.ten_iso(z_global)[..., None] * torch.eye(3, dtype=pos.dtype, device=pos.device)[None, :, :]
        ten_out = iso + sym_outer

        return {
            "scalar": self.scalar_head(z_global),
            "chiral": self.chiral_head(z_global),
            "vector": vec_out,
            "tensor": ten_out,
            "denoise": self.denoise_head(x) * atom_mask[..., None].to(pos.dtype),
            "attn": attn,
        }

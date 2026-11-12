"""Unified multi-task O(3)-equivariant 3D Science Foundation Model (YLMUniverseModel).

Unifies four symmetry ranks on one shared two-stream equivariant representation:
  1. scalar_head  (0e invariant) : energies, HOMO-LUMO gap, solvation/redox/docking scalars
  2. vector_head  (1o covariant) : dipole, atom-addressed force, force diff (T4), bond proj (T5)
  3. tensor_head  (0e+2e symm)   : quadrupole, molecular polarizability, dielectric & virial stress
  4. denoise_head (1o per-atom)  : 3D coordinate denoising / non-equilibrium score pretraining

Includes Level-of-Theory (LOT) conditioning (`wb97m`, `r2scan`, `pbe_u`, `ccsd`, `xtb`) with
strictly positive O(3)-preserving magnitude calibration and `direction_only` semi-empirical loss
masking so multi-dataset pretraining sharpens directions without blurring DFT magnitude scales.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

from src.encoder import Encoder
from src.irreps_to_text import IrrepsToText, invariants
from src.mace_backbone import MACEEncoder
from src.model import _TwoStreamStep, _has_irrep
from src.nequip_backbone import NequIPEncoder
from src.tensor_readout import TensorReadout, rotate_matrix

LOT_VOCAB: tuple[str, ...] = ("unknown", "wb97m", "r2scan", "pbe_u", "ccsd", "xtb")
LOT_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(LOT_VOCAB)}

TASK_VOCAB: tuple[str, ...] = ("scalar", "vector", "tensor", "chiral")
TASK_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(TASK_VOCAB)}

_INV_SOFTPLUS_ONE = math.log(math.e - 1.0)


def resolve_lot_id(lot: Union[str, int, None]) -> int:
    """Normalize a Level-of-Theory tag (e.g. 'wb97m', '[LOT=pbe_u]', 'GFN2-xTB') to LOT_VOCAB index."""
    if lot is None:
        return 0
    if isinstance(lot, int):
        return lot if 0 <= lot < len(LOT_VOCAB) else 0
    s = str(lot).strip().lower()
    if s.startswith("[lot=") and s.endswith("]"):
        s = s[5:-1].strip()
    if s in LOT_TO_ID:
        return LOT_TO_ID[s]
    if "wb97" in s:
        return LOT_TO_ID["wb97m"]
    if "r2scan" in s or "scan" in s:
        return LOT_TO_ID["r2scan"]
    if "pbe" in s:
        return LOT_TO_ID["pbe_u"]
    if "ccsd" in s or "mp2" in s:
        return LOT_TO_ID["ccsd"]
    if "xtb" in s or "pm6" in s or "am1" in s:
        return LOT_TO_ID["xtb"]
    return 0


def is_direction_only_lot(lot_id: torch.Tensor) -> torch.Tensor:
    """Semi-empirical rows (LOT == 'xtb') supervise directions only, never DFT magnitude scales."""
    return lot_id == LOT_TO_ID["xtb"]


def _equivariant_triple_0o(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Exact parity-odd (`0o`) Levi-Civita scalar triple product `(u x v) . w` with O(3)-invariant norm damping."""
    u_n = u / (u.pow(2).sum(dim=-1, keepdim=True) + 1.0).sqrt()
    v_n = v / (v.pow(2).sum(dim=-1, keepdim=True) + 1.0).sqrt()
    w_n = w / (w.pow(2).sum(dim=-1, keepdim=True) + 1.0).sqrt()
    return (torch.cross(u_n, v_n, dim=-1) * w_n).sum(dim=-1)


class YLMUniverseModel(nn.Module):
    """Multi-rank (0e + 1o + 2e) equivariant language-geometry foundation model."""

    def __init__(
        self,
        text_dim: int = 128,
        *,
        backbone: Union[str, nn.Module] = "e3nn",
        hidden_irreps: str = "64x0e + 8x0o + 32x1o + 8x1e + 16x2e",
        cross_irreps: str = "32x0e + 32x1o + 8x1e + 16x2e",
        n_layers: int = 3,
        n_cross_layers: int = 2,
        option: str = "C",
        c_tp_mode: str = "channelwise",
        b_score_mode: str = "normalized",
        n_elements: int = 100,
        cutoff: float = 9.45,
        n_scalars: int = 1,
        d_text: int = 256,
        n_lots: int = len(LOT_VOCAB),
    ):
        super().__init__()
        if n_cross_layers < 1:
            raise ValueError("n_cross_layers must be >= 1")
        if not (_has_irrep(cross_irreps, 0, 1) and _has_irrep(cross_irreps, 1, -1) and _has_irrep(cross_irreps, 2, 1)):
            raise ValueError("YLMUniverseModel cross_irreps must contain 0e, 1o, and 2e channels")

        self.text_dim = int(text_dim)
        self.d_text = int(d_text)
        self.n_scalars = int(n_scalars)
        self.n_cross_layers = int(n_cross_layers)
        self.option = option
        self.c_tp_mode = c_tp_mode
        self.cross_irreps = o3.Irreps(cross_irreps)

        if isinstance(backbone, nn.Module):
            self.encoder = backbone
        elif isinstance(backbone, str) and backbone.lower().startswith("mace"):
            self.encoder = MACEEncoder(backbone=backbone)
        elif isinstance(backbone, str) and backbone.lower().startswith(("nequip", "allegro")):
            self.encoder = NequIPEncoder(backbone=backbone)
        elif backbone == "e3nn":
            self.encoder = Encoder(
                hidden_irreps=hidden_irreps,
                n_layers=n_layers,
                cutoff=cutoff,
                n_elements=n_elements,
            )
        else:
            raise ValueError(
                f"Unsupported backbone {backbone!r}; expected 'e3nn', 'mace-*', 'nequip:*', 'allegro:*', or nn.Module"
            )

        self.atom_irreps = o3.Irreps(self.encoder.irreps_out)
        n_atom_0e = self.encoder.hidden_irreps.count("0e") if isinstance(self.encoder, Encoder) else 0
        self.lot_atom_embed = nn.Embedding(n_lots, max(1, n_atom_0e)) if n_atom_0e > 0 else None
        if self.lot_atom_embed is not None:
            nn.init.zeros_(self.lot_atom_embed.weight)

        self.text_proj = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, self.d_text),
            nn.SiLU(),
        )
        self.lot_text_embed = nn.Embedding(n_lots, self.d_text)
        nn.init.normal_(self.lot_text_embed.weight, std=0.02)

        self.cross = IrrepsToText(
            self.d_text,
            self.atom_irreps,
            self.cross_irreps,
            option=option,
            b_score_mode=b_score_mode,
            c_tp_mode=c_tp_mode,
        )
        if self.n_cross_layers > 1:
            self.extra_steps = nn.ModuleList([
                _TwoStreamStep(
                    self.d_text,
                    self.atom_irreps,
                    self.cross_irreps,
                    option=option,
                    b_score_mode=b_score_mode,
                    c_tp_mode=c_tp_mode,
                )
                for _ in range(self.n_cross_layers - 1)
            ])
        self.token_weight = nn.Linear(self.d_text, 1)

        # 1) 0e Invariant scalar readout (fuses text context + equivariant invariants + global atom invariants)
        scalar_in_dim = self.d_text + self.cross_irreps.num_irreps + self.atom_irreps.num_irreps
        self.scalar_head = nn.Sequential(
            nn.LayerNorm(scalar_in_dim),
            nn.Linear(scalar_in_dim, self.d_text),
            nn.SiLU(),
            nn.Linear(self.d_text, self.n_scalars),
        )
        self.lot_scalar_scale = nn.Embedding(n_lots, self.n_scalars)
        self.lot_scalar_shift = nn.Embedding(n_lots, self.n_scalars)
        nn.init.zeros_(self.lot_scalar_scale.weight)
        nn.init.zeros_(self.lot_scalar_shift.weight)

        # 2) 1o Covariant vector readout + positive LOT magnitude calibration
        self.vector_head = o3.Linear(self.cross_irreps, "1x1o")
        self.lot_vec_logscale = nn.Embedding(n_lots, 1)
        nn.init.zeros_(self.lot_vec_logscale.weight)

        # 3) 0e + 2e Symmetric 3x3 tensor readout + positive (isotropic, traceless) LOT calibration
        self.tensor_head = TensorReadout(self.cross_irreps)
        self.lot_ten_logscale = nn.Embedding(n_lots, 2)
        nn.init.zeros_(self.lot_ten_logscale.weight)

        # 4) 1o Per-atom coordinate denoising / non-equilibrium score head
        self.denoise_head = (
            o3.Linear(self.atom_irreps, "1x1o")
            if _has_irrep(self.atom_irreps, 1, -1)
            else None
        )

        # 5) 0o Parity-odd chiral pseudoscalar head (`out["chiral"]`: SO(3)-invariant, flips sign under reflection!)
        # Combines Levi-Civita `1o x 1o x 1o -> 0o` scalar triple products `(u x v) . w`, `1o x 1e -> 0o`
        # contractions, and per-atom `0o` channels gated by O(3)-invariant (`0e`) text+geometry features.
        n_chiral_chan = 0
        self.chiral_cross_u1o = o3.Linear(self.cross_irreps, "8x1o")
        self.chiral_cross_v1o = o3.Linear(self.cross_irreps, "8x1o")
        self.chiral_cross_w1o = o3.Linear(self.cross_irreps, "8x1o")
        n_chiral_chan += 8

        self._has_cross_chiral_tp = _has_irrep(self.cross_irreps, 1, -1) and _has_irrep(self.cross_irreps, 1, 1)
        if self._has_cross_chiral_tp:
            self.chiral_cross_1o = o3.Linear(self.cross_irreps, "8x1o")
            self.chiral_cross_1e = o3.Linear(self.cross_irreps, "8x1e")
            self.chiral_cross_tp = o3.ElementwiseTensorProduct("8x1o", "8x1e", ["0o"])
            n_chiral_chan += 8
        else:
            self.chiral_cross_1o = None
            self.chiral_cross_1e = None
            self.chiral_cross_tp = None

        self._has_atom_triple = sum(mul for mul, ir in self.atom_irreps if ir.l == 1 and ir.p == -1) >= 3
        if self._has_atom_triple:
            self.chiral_atom_u1o = o3.Linear(self.atom_irreps, "8x1o")
            self.chiral_atom_v1o = o3.Linear(self.atom_irreps, "8x1o")
            self.chiral_atom_w1o = o3.Linear(self.atom_irreps, "8x1o")
            n_chiral_chan += 16  # 8 atom-routed + 8 global-mean
        else:
            self.chiral_atom_u1o = None
            self.chiral_atom_v1o = None
            self.chiral_atom_w1o = None

        self._has_atom_0o = _has_irrep(self.atom_irreps, 0, -1)
        if self._has_atom_0o:
            self.chiral_atom_0o = o3.Linear(self.atom_irreps, "8x0o")
            n_chiral_chan += 16  # 8 atom-routed + 8 global-mean
        else:
            self.chiral_atom_0o = None

        self._has_atom_1o1e = _has_irrep(self.atom_irreps, 1, -1) and _has_irrep(self.atom_irreps, 1, 1)
        if self._has_atom_1o1e:
            self.chiral_atom_1o = o3.Linear(self.atom_irreps, "8x1o")
            self.chiral_atom_1e = o3.Linear(self.atom_irreps, "8x1e")
            self.chiral_atom_tp = o3.ElementwiseTensorProduct("8x1o", "8x1e", ["0o"])
            n_chiral_chan += 16  # 8 atom-routed + 8 global-mean
        else:
            self.chiral_atom_1o = None
            self.chiral_atom_1e = None
            self.chiral_atom_tp = None

        self._n_chiral_chan = n_chiral_chan
        if n_chiral_chan > 0:
            self.chiral_gate = nn.Sequential(
                nn.LayerNorm(scalar_in_dim),
                nn.Linear(scalar_in_dim, self.d_text),
                nn.SiLU(),
                nn.Linear(self.d_text, n_chiral_chan),
            )
        else:
            self.chiral_gate = None
        self.lot_chiral_logscale = nn.Embedding(n_lots, 1)
        nn.init.zeros_(self.lot_chiral_logscale.weight)

    @classmethod
    def from_mace(
        cls,
        text_dim: int = 128,
        backbone: str = "mace-mp-0-small",
        *,
        mace_model: Optional[nn.Module] = None,
        freeze_mace: bool = True,
        include_mace_forces: bool = True,
        out_irreps: Optional[str] = None,
        **kwargs,
    ) -> "YLMUniverseModel":
        """Construct YLMUniverseModel mounted on a pretrained MACE-MP-0 / MACE-OFF23 foundation backbone."""
        enc = MACEEncoder(
            backbone=backbone,
            mace_model=mace_model,
            freeze_mace=freeze_mace,
            include_mace_forces=include_mace_forces,
            out_irreps=out_irreps,
        )
        return cls(text_dim=text_dim, backbone=enc, **kwargs)

    @classmethod
    def from_nequip(
        cls,
        text_dim: int = 128,
        backbone: str = "nequip:deployed.pth",
        *,
        nequip_model: Optional[nn.Module] = None,
        atomic_numbers: Optional[Sequence[int]] = None,
        r_max: Optional[float] = None,
        node_irreps: Optional[str] = None,
        freeze_nequip: bool = True,
        include_nequip_forces: bool = True,
        out_irreps: Optional[str] = None,
        **kwargs,
    ) -> "YLMUniverseModel":
        """Construct YLMUniverseModel mounted on a compiled TorchScript or Python NequIP / Allegro backbone."""
        enc = NequIPEncoder(
            backbone=backbone,
            nequip_model=nequip_model,
            atomic_numbers=atomic_numbers,
            r_max=r_max,
            node_irreps=node_irreps,
            freeze_nequip=freeze_nequip,
            include_nequip_forces=include_nequip_forces,
            out_irreps=out_irreps,
        )
        return cls(text_dim=text_dim, backbone=enc, **kwargs)

    def encode_atoms(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        atom_mask: torch.Tensor,
        lot_id: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N = Z.shape
        if lot_id is None:
            lot_id = torch.zeros(B, dtype=torch.long, device=Z.device)
        if isinstance(self.encoder, Encoder):
            extra_sc = None
            if self.lot_atom_embed is not None:
                extra_sc = (
                    self.lot_atom_embed(lot_id)[:, None, :].expand(B, N, -1)
                    * atom_mask[..., None].to(pos.dtype)
                )
            return self.encoder(Z, pos, atom_mask, extra_scalars=extra_sc, cell=cell, pbc=pbc)
        if cell is not None:
            return self.encoder(Z, pos, atom_mask, cell=cell, pbc=pbc)
        return self.encoder(Z, pos, atom_mask)

    def predict_denoise(
        self,
        Z: torch.Tensor,
        pos_noisy: torch.Tensor,
        atom_mask: torch.Tensor,
        lot_id: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.encode_atoms(Z, pos_noisy, atom_mask, lot_id=lot_id, cell=cell, pbc=pbc)
        if self.denoise_head is None:
            return torch.zeros(Z.shape[0], Z.shape[1], 3, dtype=pos_noisy.dtype, device=pos_noisy.device)
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
        B, N = Z.shape
        if lot_id is None:
            lot_id = torch.zeros(B, dtype=torch.long, device=Z.device)

        x = self.encode_atoms(Z, pos, atom_mask, lot_id=lot_id, cell=cell, pbc=pbc)
        h = (self.text_proj(text) + self.lot_text_embed(lot_id)[:, None, :]) * text_mask[..., None].to(pos.dtype)

        e, attn = self.cross(h, x, atom_mask)
        if self.n_cross_layers > 1:
            attn_sum = attn
            for step in self.extra_steps:
                h, e, attn_step = step(h, e, x, atom_mask, text_mask)
                attn_sum = attn_sum + attn_step
            attn = attn_sum / float(self.n_cross_layers)

        has_tok = text_mask.any(dim=-1, keepdim=True)
        safe_tmask = text_mask.clone()
        safe_tmask[~has_tok.squeeze(-1), 0] = True
        w = self.token_weight(h).squeeze(-1).masked_fill(~safe_tmask, float("-inf")).softmax(dim=-1)

        h_pool = torch.einsum("bt,btd->bd", w, h)
        e_pool = torch.einsum("bt,btd->bd", w, e)
        attn_pool = torch.einsum("bt,btn->bn", w, attn)

        # 0e invariant scalar output
        inv_e = invariants(e_pool, self.cross_irreps)
        atom_count = atom_mask.sum(dim=1, keepdim=True).clamp(min=1).to(pos.dtype)
        inv_atom = (invariants(x, self.atom_irreps) * atom_mask[..., None].to(pos.dtype)).sum(dim=1) / atom_count
        raw_scalar = self.scalar_head(torch.cat([h_pool, inv_e, inv_atom], dim=-1))
        sc_scale = F.softplus(self.lot_scalar_scale(lot_id) + _INV_SOFTPLUS_ONE)
        sc_shift = self.lot_scalar_shift(lot_id)
        scalar_out = sc_scale * raw_scalar + sc_shift

        # 1o polar vector output (positive LOT scale preserves exact O(3) direction)
        vec_scale = F.softplus(self.lot_vec_logscale(lot_id) + _INV_SOFTPLUS_ONE)
        vector_out = self.vector_head(e_pool) * vec_scale

        # 0e + 2e symmetric 3x3 tensor output (decompose trace 0e + traceless 2e for exact O(3) LOT scaling)
        raw_ten = self.tensor_head(e_pool)
        eye = torch.eye(3, dtype=raw_ten.dtype, device=raw_ten.device)[None, :, :]
        iso_trace = (raw_ten.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0)[:, None, None] * eye
        aniso_traceless = raw_ten - iso_trace
        ten_scales = F.softplus(self.lot_ten_logscale(lot_id) + _INV_SOFTPLUS_ONE)
        s_iso = ten_scales[:, 0:1, None]
        s_aniso = ten_scales[:, 1:2, None]
        tensor_out = s_iso * iso_trace + s_aniso * aniso_traceless

        denoise_out = (
            self.denoise_head(x) * atom_mask[..., None].to(pos.dtype)
            if self.denoise_head is not None
            else torch.zeros(B, N, 3, dtype=pos.dtype, device=pos.device)
        )

        # 0o chiral pseudoscalar output (`chiral(pos @ Q.T) == det(Q) * chiral(pos)`)
        if self._n_chiral_chan > 0 and self.chiral_gate is not None:
            chiral_parts: list[torch.Tensor] = []
            mask_f = atom_mask[..., None].to(pos.dtype)
            cu = self.chiral_cross_u1o(e_pool).reshape(B, 8, 3)
            cv = self.chiral_cross_v1o(e_pool).reshape(B, 8, 3)
            cw = self.chiral_cross_w1o(e_pool).reshape(B, 8, 3)
            chiral_parts.append(_equivariant_triple_0o(cu, cv, cw))
            if self._has_cross_chiral_tp:
                c_1o = self.chiral_cross_1o(e_pool)
                c_1e = self.chiral_cross_1e(e_pool)
                chiral_parts.append(self.chiral_cross_tp(c_1o, c_1e))
            if self._has_atom_triple:
                au = self.chiral_atom_u1o(x).reshape(B, N, 8, 3)
                av = self.chiral_atom_v1o(x).reshape(B, N, 8, 3)
                aw = self.chiral_atom_w1o(x).reshape(B, N, 8, 3)
                a_tri = _equivariant_triple_0o(au, av, aw) * mask_f
                chiral_parts.append(torch.einsum("bn,bnd->bd", attn_pool, a_tri))
                chiral_parts.append(a_tri.sum(dim=1) / atom_count)
            if self._has_atom_0o:
                a_0o = self.chiral_atom_0o(x) * mask_f
                chiral_parts.append(torch.einsum("bn,bnd->bd", attn_pool, a_0o))
                chiral_parts.append(a_0o.sum(dim=1) / atom_count)
            if self._has_atom_1o1e:
                a_tp = self.chiral_atom_tp(self.chiral_atom_1o(x), self.chiral_atom_1e(x)) * mask_f
                chiral_parts.append(torch.einsum("bn,bnd->bd", attn_pool, a_tp))
                chiral_parts.append(a_tp.sum(dim=1) / atom_count)
            chiral_0o = torch.cat(chiral_parts, dim=-1)
            gate_0e = self.chiral_gate(torch.cat([h_pool, inv_e, inv_atom], dim=-1))
            ch_scale = F.softplus(self.lot_chiral_logscale(lot_id) + _INV_SOFTPLUS_ONE)
            chiral_out = ch_scale * (gate_0e * chiral_0o).sum(dim=-1, keepdim=True)
        else:
            chiral_out = torch.zeros(B, 1, dtype=pos.dtype, device=pos.device)

        return {
            "scalar": scalar_out,
            "chiral": chiral_out,
            "vector": vector_out,
            "tensor": tensor_out,
            "denoise": denoise_out,
            "attn": attn_pool,
            "e_pool": e_pool,
            "h_pool": h_pool,
        }


def decoupled_vector_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    direction_only: Optional[torch.Tensor] = None,
    lambda_dir: float = 1.0,
    lambda_mag: float = 0.5,
    lambda_huber: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Decoupled direction (1 - cos) + magnitude L1 + Huber loss with semi-empirical direction_only masking."""
    if pred.numel() == 0:
        return pred.sum() * 0.0
    p_norm = (pred.pow(2).sum(dim=-1) + eps * eps).sqrt()
    t_norm = target.norm(dim=-1)
    nonzero = t_norm > 1e-6

    dot = (pred * target).sum(dim=-1)
    cos = dot / (p_norm * t_norm.clamp(min=eps))
    l_dir = torch.where(nonzero, 1.0 - cos.clamp(-1.0, 1.0), torch.zeros_like(t_norm))

    l_mag = (p_norm - t_norm).abs()
    l_huber = F.huber_loss(pred, target, reduction="none", delta=1.0).mean(dim=-1)

    if direction_only is not None:
        dir_only = direction_only.to(dtype=torch.bool, device=pred.device)
        # Semi-empirical rows supervise pure direction when |target| > 0; zero-dipole symmetry rows still damp norm.
        full_row = lambda_dir * l_dir + lambda_mag * l_mag + lambda_huber * l_huber
        dir_row = torch.where(nonzero, lambda_dir * l_dir, lambda_mag * l_mag)
        row_loss = torch.where(dir_only, dir_row, full_row)
    else:
        row_loss = lambda_dir * l_dir + lambda_mag * l_mag + lambda_huber * l_huber
    return row_loss.mean()


def symmetric_tensor_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    direction_only: Optional[torch.Tensor] = None,
    lambda_dir: float = 0.5,
    lambda_frob: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Strictly O(3)-invariant Frobenius cosine + Huber loss on 3x3 symmetric tensors."""
    if pred.numel() == 0:
        return pred.sum() * 0.0
    p_flat = pred.reshape(pred.shape[0], 9)
    t_flat = target.reshape(target.shape[0], 9)
    p_norm = (p_flat.pow(2).sum(dim=-1) + eps * eps).sqrt()
    t_norm = t_flat.norm(dim=-1)
    nonzero = t_norm > 1e-6

    cos = (p_flat * t_flat).sum(dim=-1) / (p_norm * t_norm.clamp(min=eps))
    l_dir = torch.where(nonzero, 1.0 - cos.clamp(-1.0, 1.0), torch.zeros_like(t_norm))
    l_frob = F.huber_loss(p_flat, t_flat, reduction="none", delta=1.0).mean(dim=-1)

    if direction_only is not None:
        dir_only = direction_only.to(dtype=torch.bool, device=pred.device)
        row_loss = torch.where(dir_only & nonzero, lambda_dir * l_dir, lambda_dir * l_dir + lambda_frob * l_frob)
    else:
        row_loss = lambda_dir * l_dir + lambda_frob * l_frob
    return row_loss.mean()


def routing_supervision_loss(
    attn_pool: torch.Tensor,
    target_atom_mask: Optional[torch.Tensor],
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Supervise pooled language->atom routing mass onto queried target atom(s) (T3 single atom, T4/T5 atom pair)."""
    if target_atom_mask is None or attn_pool.numel() == 0:
        return attn_pool.sum() * 0.0
    tmask = target_atom_mask.to(dtype=torch.bool, device=attn_pool.device)
    active = tmask.any(dim=-1)
    if not active.any():
        return attn_pool.sum() * 0.0
    mass = (attn_pool * tmask.to(attn_pool.dtype)).sum(dim=-1)
    nll = -torch.log(mass[active].clamp(min=eps, max=1.0))
    return nll.mean()


def coordinate_denoise_loss(
    model: nn.Module,
    Z: torch.Tensor,
    pos: torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    lot_id: Optional[torch.Tensor] = None,
    cell: Optional[torch.Tensor] = None,
    pbc: Optional[torch.Tensor] = None,
    noise_std: float = 0.12,
) -> torch.Tensor:
    """Zero-CoM 3D coordinate denoising pretraining objective (1o score matching)."""
    if not hasattr(model, "predict_denoise"):
        return pos.sum() * 0.0
    m = atom_mask[..., None].to(pos.dtype)
    eps = torch.randn_like(pos) * m
    counts = atom_mask.sum(dim=1, keepdim=True).clamp(min=1).to(pos.dtype)[..., None]
    eps = (eps - eps.sum(dim=1, keepdim=True) / counts) * m
    pos_noisy = pos + noise_std * eps
    pred_eps = model.predict_denoise(Z, pos_noisy, atom_mask, lot_id=lot_id, cell=cell, pbc=pbc)
    diff2 = ((pred_eps - eps) * m).pow(2).sum(dim=-1)
    denom = atom_mask.sum().clamp(min=1).to(pos.dtype)
    return diff2.sum() / denom


def compute_universe_loss(
    outputs: dict[str, torch.Tensor],
    *,
    task_kind: torch.Tensor,
    scalar_target: torch.Tensor,
    vector_target: torch.Tensor,
    tensor_target: torch.Tensor,
    target_atom_mask: Optional[torch.Tensor] = None,
    lot_id: Optional[torch.Tensor] = None,
    attn_loss_weight: float = 0.2,
    lambda_dir: float = 1.0,
    lambda_mag: float = 0.5,
    lambda_huber: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Multi-task Universe loss combining 0e scalar, 1o vector, 0e+2e tensor, and atom-routing objectives."""
    zero = outputs["vector"].sum() * 0.0
    dir_only = is_direction_only_lot(lot_id) if lot_id is not None else None

    is_sc = task_kind == TASK_TO_ID["scalar"]
    is_vec = task_kind == TASK_TO_ID["vector"]
    is_ten = task_kind == TASK_TO_ID["tensor"]
    is_ch = task_kind == TASK_TO_ID["chiral"]

    sc_mask = is_sc & (~dir_only if dir_only is not None else is_sc)
    if sc_mask.any():
        pred_sc = outputs["scalar"][sc_mask].reshape(-1)
        targ_sc = scalar_target[sc_mask].reshape(-1)
        l_scalar = F.huber_loss(pred_sc, targ_sc, delta=1.0)
    else:
        l_scalar = zero

    if is_ch.any() and "chiral" in outputs:
        pred_ch = outputs["chiral"][is_ch].reshape(-1)
        targ_ch = scalar_target[is_ch].reshape(-1)
        nz_ch = targ_ch.abs() > 1e-6
        sign_term = (
            F.softplus(-torch.sign(targ_ch[nz_ch]) * pred_ch[nz_ch]).mean()
            if nz_ch.any()
            else zero
        )
        l_chiral = 0.5 * sign_term + 0.5 * F.huber_loss(pred_ch, targ_ch, delta=1.0)
    else:
        l_chiral = zero

    if is_vec.any():
        l_vector = decoupled_vector_loss(
            outputs["vector"][is_vec],
            vector_target[is_vec],
            direction_only=None if dir_only is None else dir_only[is_vec],
            lambda_dir=lambda_dir,
            lambda_mag=lambda_mag,
            lambda_huber=lambda_huber,
        )
    else:
        l_vector = zero

    if is_ten.any():
        l_tensor = symmetric_tensor_loss(
            outputs["tensor"][is_ten],
            tensor_target[is_ten],
            direction_only=None if dir_only is None else dir_only[is_ten],
        )
    else:
        l_tensor = zero

    l_route = routing_supervision_loss(outputs["attn"], target_atom_mask)

    active_terms = [
        t
        for flag, t in (
            (sc_mask.any(), l_scalar),
            (is_ch.any() and "chiral" in outputs, l_chiral),
            (is_vec.any(), l_vector),
            (is_ten.any(), l_tensor),
        )
        if flag
    ]
    total = (sum(active_terms) if active_terms else zero) + attn_loss_weight * l_route
    return {
        "loss": total,
        "scalar_loss": l_scalar,
        "chiral_loss": l_chiral,
        "vector_loss": l_vector,
        "tensor_loss": l_tensor,
        "routing_loss": l_route,
    }


@torch.no_grad()
def evaluate_universe_o3(
    model: nn.Module,
    batches: Sequence[dict[str, torch.Tensor]],
    *,
    n_rotations: int = 8,
    include_reflections: bool = True,
    seed: int = 123,
    device: Union[str, torch.device] = "cpu",
) -> dict[str, float]:
    """Evaluate a multi-head model under K random O(3) rotations & reflections on scalar, vector, tensor, and routing."""
    was_training = model.training
    model.eval()

    rotations: list[torch.Tensor] = [torch.eye(3, dtype=torch.float32, device=device)]
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for k in range(max(0, n_rotations - 1)):
            R = o3.rand_matrix().to(dtype=torch.float32, device=device)
            if include_reflections and (k % 2 == 1):
                R = -R
            rotations.append(R)

    sc_abs_err: list[float] = []
    sc_inv_err: list[float] = []
    ch_sign_hits: list[float] = []
    ch_abs_err: list[float] = []
    ch_parity_err: list[float] = []
    vec_cos: list[float] = []
    vec_mae: list[float] = []
    vec_mag_mae: list[float] = []
    vec_eq_err: list[float] = []
    subtask_cos: dict[str, list[float]] = {}
    ten_cos: list[float] = []
    ten_mae: list[float] = []
    ten_eq_err: list[float] = []
    route_hits: list[float] = []
    route_mass: list[float] = []

    for batch in batches:
        Z = batch["Z"].to(device)
        pos0 = batch["pos"].to(device)
        atom_mask = batch["atom_mask"].to(device)
        text = batch["text"].to(device)
        text_mask = batch["text_mask"].to(device)
        lot_id = batch["lot_id"].to(device)
        task_kind = batch["task_kind"].to(device)
        sc_targ = batch["scalar_target"].to(device)
        vec_targ0 = batch["vector_target"].to(device)
        ten_targ0 = batch["tensor_target"].to(device)
        target_atom_mask = batch.get("target_atom_mask")
        if target_atom_mask is not None:
            target_atom_mask = target_atom_mask.to(device)
        cell0 = batch.get("cell")
        pbc = batch.get("pbc")
        if cell0 is not None and pbc is not None and pbc.any():
            cell0 = cell0.to(device)
            pbc = pbc.to(device)
        else:
            cell0 = None
            pbc = None
        subtasks: Optional[list[str]] = batch.get("subtasks")

        out0 = model(Z, pos0, atom_mask, text, text_mask, lot_id=lot_id, cell=cell0, pbc=pbc)

        if target_atom_mask is not None:
            has_t = target_atom_mask.any(dim=-1)
            if has_t.any():
                attn_sub = out0["attn"][has_t]
                tmask_sub = target_atom_mask[has_t]
                top1 = attn_sub.argmax(dim=-1)
                hits = tmask_sub[torch.arange(top1.shape[0], device=device), top1].float()
                mass = (attn_sub * tmask_sub.float()).sum(dim=-1)
                route_hits.extend(hits.cpu().tolist())
                route_mass.extend(mass.cpu().tolist())

        for r_idx, R in enumerate(rotations):
            pos_r = pos0 @ R.T
            cell_r = None if cell0 is None else cell0 @ R.T
            out_r = out0 if r_idx == 0 else model(
                Z, pos_r, atom_mask, text, text_mask, lot_id=lot_id, cell=cell_r, pbc=pbc
            )

            is_sc = task_kind == TASK_TO_ID["scalar"]
            if is_sc.any():
                pred_s = out_r["scalar"][is_sc].reshape(-1)
                targ_s = sc_targ[is_sc].reshape(-1)
                sc_abs_err.extend((pred_s - targ_s).abs().cpu().tolist())
                if r_idx > 0:
                    s0 = out0["scalar"][is_sc].reshape(-1)
                    sc_inv_err.extend((pred_s - s0).abs().cpu().tolist())

            is_ch = task_kind == TASK_TO_ID["chiral"]
            if is_ch.any() and "chiral" in out_r:
                det_sign = 1.0 if float(torch.linalg.det(R).item()) >= 0.0 else -1.0
                pred_c = out_r["chiral"][is_ch].reshape(-1)
                targ_c = sc_targ[is_ch].reshape(-1) * det_sign
                ch_abs_err.extend((pred_c - targ_c).abs().cpu().tolist())
                nz_c = targ_c.abs() > 1e-6
                if nz_c.any():
                    # Strict sign match (> 0 for + target, < 0 for - target; 0 counts as miss!)
                    strict_match = ((pred_c[nz_c] * targ_c[nz_c]) > 1e-8).float()
                    ch_sign_hits.extend(strict_match.cpu().tolist())
                if r_idx > 0 and "chiral" in out0:
                    c0_rot = out0["chiral"][is_ch].reshape(-1) * det_sign
                    ch_parity_err.extend((pred_c - c0_rot).abs().cpu().tolist())

            is_vec = task_kind == TASK_TO_ID["vector"]
            if is_vec.any():
                pv = out_r["vector"][is_vec]
                tv = (vec_targ0 @ R.T)[is_vec]
                t_norm = tv.norm(dim=-1)
                p_norm = pv.norm(dim=-1)
                nz = t_norm > 1e-6
                if nz.any():
                    cos_v = (pv[nz] * tv[nz]).sum(dim=-1) / (p_norm[nz].clamp(min=1e-8) * t_norm[nz])
                    cos_list = cos_v.cpu().tolist()
                    vec_cos.extend(cos_list)
                    if subtasks is not None:
                        vec_indices = is_vec.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
                        nz_flags = nz.cpu().tolist()
                        for row_i, keep_nz in zip(vec_indices, nz_flags):
                            if keep_nz:
                                st = subtasks[row_i]
                                c_val = float(
                                    ((out_r["vector"][row_i] * (vec_targ0[row_i] @ R.T)).sum()
                                     / (out_r["vector"][row_i].norm().clamp(min=1e-8) * (vec_targ0[row_i] @ R.T).norm().clamp(min=1e-8))).cpu()
                                )
                                subtask_cos.setdefault(st, []).append(c_val)
                vec_mae.extend((pv - tv).abs().mean(dim=-1).cpu().tolist())
                vec_mag_mae.extend((p_norm - t_norm).abs().cpu().tolist())
                if r_idx > 0:
                    v0_rot = out0["vector"][is_vec] @ R.T
                    eq_v = (pv - v0_rot).norm(dim=-1) / out0["vector"][is_vec].norm(dim=-1).clamp(min=1e-8)
                    vec_eq_err.extend(eq_v.cpu().tolist())

            is_ten = task_kind == TASK_TO_ID["tensor"]
            if is_ten.any():
                pt = out_r["tensor"][is_ten]
                tt = rotate_matrix(ten_targ0[is_ten], R)
                pf = pt.reshape(pt.shape[0], 9)
                tf = tt.reshape(tt.shape[0], 9)
                tn = tf.norm(dim=-1)
                pn = pf.norm(dim=-1)
                nz_t = tn > 1e-6
                if nz_t.any():
                    ct = (pf[nz_t] * tf[nz_t]).sum(dim=-1) / (pn[nz_t].clamp(min=1e-8) * tn[nz_t])
                    ten_cos.extend(ct.cpu().tolist())
                ten_mae.extend((pf - tf).abs().mean(dim=-1).cpu().tolist())
                if r_idx > 0:
                    t0_rot = rotate_matrix(out0["tensor"][is_ten], R)
                    eq_t = (
                        (pt - t0_rot).reshape(pt.shape[0], 9).norm(dim=-1)
                        / out0["tensor"][is_ten].reshape(pt.shape[0], 9).norm(dim=-1).clamp(min=1e-8)
                    )
                    ten_eq_err.extend(eq_t.cpu().tolist())

    if was_training:
        model.train()

    def _mean(vals: list[float], default: float = 0.0) -> float:
        return float(sum(vals) / len(vals)) if vals else default

    metrics: dict[str, float] = {
        "scalar_mae": _mean(sc_abs_err),
        "scalar_inv_err": _mean(sc_inv_err),
        "chiral_sign_acc": _mean(ch_sign_hits),
        "chiral_mae": _mean(ch_abs_err),
        "chiral_parity_err": _mean(ch_parity_err),
        "vector_cos": _mean(vec_cos),
        "vector_mae": _mean(vec_mae),
        "vector_mag_mae": _mean(vec_mag_mae),
        "vector_eq_err": _mean(vec_eq_err),
        "tensor_cos": _mean(ten_cos),
        "tensor_mae": _mean(ten_mae),
        "tensor_eq_err": _mean(ten_eq_err),
        "routing_acc": _mean(route_hits),
        "routing_mass": _mean(route_mass),
    }
    for st_name, vals in sorted(subtask_cos.items()):
        metrics[f"{st_name}_cos"] = _mean(vals)
    return metrics

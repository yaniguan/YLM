"""Adapter that plugs pretrained MACE models (`MACE-MP-0` / `MACE-OFF23`) into `YLM`.

Passes MACE's 3D atom features and exact forces (`F = -dE/dx`) into YLM's text-to-3D attention
so rotating or mirroring the atoms rotates or mirrors the output arrows the exact same way.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import warnings

import torch
import torch.nn as nn
from e3nn import o3

from src.encoder import _periodic_edges

BOHR_TO_ANGSTROM = 0.529177210903
AU_FORCE_TO_EV_ANG = 51.4220674763  # 1 Hartree / Bohr in eV / Angstrom

# Periodic-table main-group / transition-metal fallback groups for out-of-vocabulary elements
_VALENCE_GROUPS = [
    (1, 3, 11, 19, 37, 55),
    (4, 12, 20, 38, 56),
    (5, 13, 31, 49),
    (6, 14, 32, 50, 82),
    (7, 15, 33, 51, 83),
    (8, 16, 34, 52),
    (9, 17, 35, 53, 85),
    (2, 10, 18, 36, 54),
    (21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 40, 41, 42, 44, 45, 46, 47, 48, 74, 78, 79),
]


def load_mace_foundation_model(
    backbone: Union[str, Path] = "mace-mp-0-small",
    device: Union[str, torch.device] = "cpu",
    default_dtype: str = "float32",
) -> nn.Module:
    """Load a pretrained MACE-MP-0 / MACE-OFF23 foundation model or a local MACE `.model` file."""
    spec = str(backbone).strip()
    if spec.startswith("mace:"):
        spec_path = Path(spec[5:])
    else:
        spec_path = Path(spec)

    if spec_path.exists():
        loaded = torch.load(str(spec_path), map_location=device, weights_only=False)
        if hasattr(loaded, "models"):
            return loaded.models[0]
        if isinstance(loaded, nn.Module):
            return loaded
        raise ValueError(f"Loaded object from {spec_path} is not a PyTorch MACE module.")

    low = spec.lower()
    try:
        from mace.calculators import mace_mp, mace_off
    except ImportError as exc:
        raise ImportError(
            f"Cannot load pretrained backbone {spec!r} because 'mace-torch' is not installed. "
            "Install it with `pip install mace-torch` or pass an in-memory `mace_model`."
        ) from exc

    dev_str = str(device)
    if low.startswith("mace-mp"):
        # Examples: "mace-mp-0" -> "small", "mace-mp-0-medium" -> "medium", "mace-mp-0-large" -> "large"
        size = "small"
        for tag in ("small", "medium", "large", "medium-mpa-0", "medium-omat-0"):
            if low.endswith(tag):
                size = tag
                break
        calc = mace_mp(model=size, device=dev_str, default_dtype=default_dtype)
        return calc.models[0]

    if low.startswith("mace-off"):
        size = "small"
        for tag in ("small", "medium", "large"):
            if low.endswith(tag):
                size = tag
                break
        calc = mace_off(model=size, device=dev_str, default_dtype=default_dtype)
        return calc.models[0]

    raise ValueError(
        f"Unrecognized MACE backbone {spec!r}. Expected 'mace-mp-0[-small|-medium|-large]', "
        "'mace-off23[-small|-medium|-large]', or 'mace:/path/to/checkpoint.model'."
    )


def infer_mace_node_irreps(mace_model: nn.Module) -> o3.Irreps:
    """Infer exact `e3nn.o3.Irreps` of `out['node_feats']` from a MACE module."""
    if hasattr(mace_model, "node_feats_irreps"):
        return o3.Irreps(mace_model.node_feats_irreps)
    if hasattr(mace_model, "products"):
        parts = []
        for prod in mace_model.products:
            if hasattr(prod, "linear") and hasattr(prod.linear, "irreps_out"):
                parts.append(str(o3.Irreps(prod.linear.irreps_out)))
            elif hasattr(prod, "irreps_out"):
                parts.append(str(o3.Irreps(prod.irreps_out)))
        if parts:
            return o3.Irreps(" + ".join(parts))
    raise ValueError(
        "Could not infer `node_feats` irreps from `mace_model` (expected `.products[i].linear.irreps_out` "
        "or `.node_feats_irreps`)."
    )


def _scalar_slice_indices(irreps: o3.Irreps) -> list[int]:
    """Return flat feature indices corresponding to `0e` scalar channels."""
    idx: list[int] = []
    offset = 0
    for mul, ir in irreps:
        dim = mul * ir.dim
        if ir.l == 0 and ir.p == 1:
            idx.extend(range(offset, offset + dim))
        offset += dim
    return idx


def _nearest_supported_z(z: int, supported: list[int]) -> int:
    if z in supported:
        return z
    sup_set = set(supported)
    for grp in _VALENCE_GROUPS:
        if z in grp:
            for cand in grp:
                if cand in sup_set:
                    return cand
    return min(supported, key=lambda s: abs(s - z))


class MACEEncoder(nn.Module):
    """Adapter turning a pretrained MACE module into an O(3)-equivariant YLM `Encoder` drop-in.

    Inputs follow YLM's native convention (`pos` and `cell` in Bohr) and are converted internally
    to Angstrom for MACE, while conservative MACE forces (`eV/Å`) are converted back to atomic
    units (`Hartree/Bohr`) and concatenated as an exact `1x1o` equivariant channel.
    """

    def __init__(
        self,
        backbone: Union[str, Path] = "mace-mp-0-small",
        mace_model: Optional[nn.Module] = None,
        *,
        freeze_mace: bool = True,
        include_mace_forces: bool = True,
        out_irreps: Optional[str] = None,
        n_motion_vectors: int = 0,
        default_dtype: str = "float32",
    ):
        super().__init__()
        if mace_model is None:
            mace_model = load_mace_foundation_model(backbone=backbone, default_dtype=default_dtype)
        self.backbone_name = str(backbone)
        self.mace = mace_model
        self.freeze_mace = bool(freeze_mace)
        self.include_mace_forces = bool(include_mace_forces)
        self.n_motion_vectors = int(n_motion_vectors)

        if self.freeze_mace:
            self.mace.eval()
            for p in self.mace.parameters():
                p.requires_grad_(False)

        if not hasattr(self.mace, "atomic_numbers"):
            raise ValueError("`mace_model` must expose `.atomic_numbers` tensor/sequence.")
        z_list = [int(z) for z in torch.as_tensor(self.mace.atomic_numbers).tolist()]
        self.supported_atomic_numbers = z_list
        self.num_species = len(z_list)

        r_max_ang = float(torch.as_tensor(getattr(self.mace, "r_max", 5.0)).item())
        self.r_max_ang = r_max_ang
        self.cutoff = r_max_ang / BOHR_TO_ANGSTROM

        # Precompute lookup table Z in [0..118] -> species index in [0..num_species-1]
        z_to_idx = torch.zeros(119, dtype=torch.long)
        exact_mask = torch.zeros(119, dtype=torch.bool)
        exact_map = {z: i for i, z in enumerate(z_list)}
        for z in range(1, 119):
            if z in exact_map:
                z_to_idx[z] = exact_map[z]
                exact_mask[z] = True
            else:
                mapped = _nearest_supported_z(z, z_list)
                z_to_idx[z] = exact_map[mapped]
        self.register_buffer("_z_to_species", z_to_idx, persistent=False)
        self.register_buffer("_z_is_exact", exact_mask, persistent=False)
        self.last_element_warnings: list[str] = []

        self.node_feats_irreps = infer_mace_node_irreps(self.mace)
        raw_parts = [str(self.node_feats_irreps)]
        if self.include_mace_forces:
            raw_parts.append("1x1o")
        if self.n_motion_vectors > 0:
            raw_parts.append(f"{self.n_motion_vectors}x1o")
        self.raw_irreps = o3.Irreps(" + ".join(raw_parts))

        scalar_idx = _scalar_slice_indices(self.node_feats_irreps)
        self.n_scalar = len(scalar_idx)
        self.register_buffer("_scalar_idx", torch.tensor(scalar_idx, dtype=torch.long), persistent=False)

        if out_irreps is not None:
            self.proj: Optional[nn.Module] = o3.Linear(self.raw_irreps, o3.Irreps(out_irreps))
            self.irreps_out = o3.Irreps(out_irreps)
        else:
            self.proj = None
            self.irreps_out = self.raw_irreps
        # Keep `.hidden_irreps` alias so `IrrepsTokenizer` can query `.hidden_irreps.count("0e")`
        self.hidden_irreps = self.irreps_out

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_mace:
            self.mace.eval()
        return self

    def _mace_dtype(self, fallback: torch.dtype) -> torch.dtype:
        for p in self.mace.parameters():
            return p.dtype
        for b in self.mace.buffers():
            if b.is_floating_point():
                return b.dtype
        return fallback

    def _build_mace_batch(
        self,
        Z: torch.Tensor,
        pos_ang: torch.Tensor,
        mask: torch.Tensor,
        cell_ang: Optional[torch.Tensor],
        pbc: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Convert padded `(B, N, ...)` tensors into a PyG-style MACE `AtomicData` batch dict."""
        device = pos_ang.device
        B, N = Z.shape

        flat_to_compact = torch.full((B * N,), -1, dtype=torch.long, device=device)
        positions_list: list[torch.Tensor] = []
        z_compact_list: list[torch.Tensor] = []
        batch_list: list[torch.Tensor] = []
        ptr = [0]
        cells_list: list[torch.Tensor] = []
        valid_b_indices: list[int] = []

        running = 0
        for b in range(B):
            idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            m = idx.numel()
            if m == 0:
                continue
            g_id = len(valid_b_indices)
            valid_b_indices.append(b)
            flat_idx = b * N + idx
            flat_to_compact[flat_idx] = torch.arange(running, running + m, device=device, dtype=torch.long)
            positions_list.append(pos_ang[b, idx].to(dtype))
            z_compact_list.append(Z[b, idx])
            batch_list.append(torch.full((m,), g_id, dtype=torch.long, device=device))
            if cell_ang is not None:
                c_b = cell_ang[b] if cell_ang.ndim == 3 else cell_ang
                cells_list.append(c_b.to(dtype))
            else:
                cells_list.append(torch.zeros((3, 3), dtype=dtype, device=device))
            running += m
            ptr.append(running)

        if running == 0:
            empty_data: dict[str, torch.Tensor] = {}
            return empty_data, flat_to_compact, torch.empty(0, dtype=torch.long, device=device)

        positions = torch.cat(positions_list, dim=0)
        z_compact = torch.cat(z_compact_list, dim=0).clamp(min=1, max=118)

        # Record warnings for any atomic numbers outside the MACE model's native vocabulary
        inexact = z_compact[~self._z_is_exact[z_compact]]
        self.last_element_warnings = []
        if inexact.numel() > 0:
            uniq = sorted(set(int(v) for v in inexact.tolist()))
            for uz in uniq:
                mapped_z = self.supported_atomic_numbers[int(self._z_to_species[uz].item())]
                msg = f"Z={uz} not in {self.backbone_name} species; mapped to valence analogue Z={mapped_z}."
                self.last_element_warnings.append(msg)
            warnings.warn("; ".join(self.last_element_warnings), UserWarning, stacklevel=2)

        sp_idx = self._z_to_species[z_compact]
        node_attrs = torch.nn.functional.one_hot(sp_idx, num_classes=self.num_species).to(dtype)
        batch_vec = torch.cat(batch_list, dim=0)
        ptr_vec = torch.tensor(ptr, dtype=torch.long, device=device)
        cell_cat = torch.cat(cells_list, dim=0)  # (n_graphs * 3, 3)

        if cell_ang is None:
            rel = pos_ang[:, :, None, :] - pos_ang[:, None, :, :]
            dist = rel.norm(dim=-1)
            pair = mask[:, :, None] & mask[:, None, :] & (dist < self.r_max_ang)
            pair &= ~torch.eye(N, dtype=torch.bool, device=device)[None]
            b_idx, i_idx, j_idx = pair.nonzero(as_tuple=True)
            dst_flat = b_idx * N + i_idx
            src_flat = b_idx * N + j_idx
            dst_c = flat_to_compact[dst_flat]
            src_c = flat_to_compact[src_flat]
            edge_index = torch.stack([src_c, dst_c], dim=0)
            shifts = torch.zeros((dst_c.numel(), 3), dtype=dtype, device=device)
            unit_shifts = torch.zeros((dst_c.numel(), 3), dtype=dtype, device=device)
        else:
            dst_flat, src_flat, vec_ang, _ = _periodic_edges(pos_ang, mask, cell_ang, pbc, self.r_max_ang)
            dst_c = flat_to_compact[dst_flat]
            src_c = flat_to_compact[src_flat]
            edge_index = torch.stack([src_c, dst_c], dim=0)
            # MACE convention: vectors = positions[dst] - positions[src] + shifts == vec_ang
            shifts = (vec_ang.to(dtype) - (positions[dst_c] - positions[src_c])).detach()
            unit_shifts = torch.zeros_like(shifts)

        n_graphs = len(valid_b_indices)
        data = {
            "positions": positions,
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell_cat,
            "batch": batch_vec,
            "ptr": ptr_vec,
            "weight": torch.ones(n_graphs, dtype=dtype, device=device),
            "head": torch.zeros(n_graphs, dtype=torch.long, device=device),
        }
        compact_to_flat = (flat_to_compact >= 0).nonzero(as_tuple=False).squeeze(-1)
        # Ensure compact_to_flat is ordered by compact index 0..M-1
        order = torch.argsort(flat_to_compact[compact_to_flat])
        compact_to_flat = compact_to_flat[order]
        return data, flat_to_compact, compact_to_flat

    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor,
        extra_scalars: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
        extra_vectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return `(B, N, irreps_out.dim)` O(3)-equivariant atom features."""
        B, N = Z.shape
        out_dtype = pos.dtype if pos.is_floating_point() else torch.get_default_dtype()
        mace_dtype = self._mace_dtype(out_dtype)

        pos_ang = pos.to(mace_dtype) * BOHR_TO_ANGSTROM
        cell_ang = cell.to(mace_dtype) * BOHR_TO_ANGSTROM if cell is not None else None

        data, _, compact_to_flat = self._build_mace_batch(Z, pos_ang, mask, cell_ang, pbc, mace_dtype)
        if compact_to_flat.numel() == 0:
            return torch.zeros((B, N, self.irreps_out.dim), dtype=out_dtype, device=pos.device)

        need_grad_for_train = torch.is_grad_enabled() and (not self.freeze_mace or pos.requires_grad)
        if self.include_mace_forces:
            # Conservative force computation (-dE/dr) inside MACE requires autograd even during inference
            with torch.enable_grad():
                if not need_grad_for_train:
                    data["positions"] = data["positions"].detach().requires_grad_(True)
                elif not data["positions"].requires_grad:
                    data["positions"] = data["positions"].requires_grad_(True)
                mace_out = self.mace(
                    data,
                    training=(self.training and not self.freeze_mace),
                    compute_force=True,
                )
        else:
            if self.freeze_mace and not pos.requires_grad:
                with torch.no_grad():
                    mace_out = self.mace(data, training=False, compute_force=False)
            else:
                mace_out = self.mace(
                    data,
                    training=(self.training and not self.freeze_mace),
                    compute_force=False,
                )

        node_feats = mace_out["node_feats"]
        if not need_grad_for_train:
            node_feats = node_feats.detach()
        node_feats = node_feats.to(out_dtype)

        feats_parts = [node_feats]
        if self.include_mace_forces:
            forces_ev_ang = mace_out["forces"]
            if not need_grad_for_train:
                forces_ev_ang = forces_ev_ang.detach()
            forces_au = (forces_ev_ang / AU_FORCE_TO_EV_ANG).to(out_dtype)
            feats_parts.append(forces_au)

        compact_raw = torch.cat(feats_parts, dim=-1)
        raw_no_motion_dim = compact_raw.shape[-1]
        full_grid = torch.zeros((B * N, raw_no_motion_dim), dtype=out_dtype, device=pos.device)
        full_grid[compact_to_flat] = compact_raw

        if extra_scalars is not None and self._scalar_idx.numel() > 0:
            es = extra_scalars.reshape(B * N, -1).to(out_dtype) * mask.reshape(B * N, 1).to(out_dtype)
            k = min(es.shape[-1], self._scalar_idx.numel())
            idx_s = self._scalar_idx[:k]
            full_grid = full_grid.clone()
            full_grid[:, idx_s] = full_grid[:, idx_s] + es[:, :k]

        if self.n_motion_vectors > 0:
            if extra_vectors is None:
                v = torch.zeros((B * N, self.n_motion_vectors * 3), dtype=out_dtype, device=pos.device)
            else:
                v = extra_vectors.to(device=pos.device, dtype=out_dtype).reshape(B * N, self.n_motion_vectors * 3)
                v = v * mask.reshape(B * N, 1).to(out_dtype)
            full_grid = torch.cat([full_grid, v], dim=-1)

        if self.proj is not None:
            proj_dtype = next(self.proj.parameters()).dtype
            full_grid = self.proj(full_grid.to(proj_dtype)).to(out_dtype)

        return full_grid.reshape(B, N, -1) * mask[..., None].to(out_dtype)

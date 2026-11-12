"""Adapter that plugs NequIP and Allegro `.pth` models into `YLM` and live MD runners.

Loads compiled TorchScript (`nequip-deploy` / `allegro`) `.pth` files or Python `nequip` models
and passes their 3D atom features and forces (`F = -dE/dx`) into YLM without losing 3D symmetry.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Optional, Sequence, Union
import warnings

import torch
import torch.nn as nn
from e3nn import o3

from src.encoder import _periodic_edges
from src.mace_backbone import (
    AU_FORCE_TO_EV_ANG,
    BOHR_TO_ANGSTROM,
    _nearest_supported_z,
    _scalar_slice_indices,
)

_PERIODIC_SYMBOLS = (
    "",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
_SYM_TO_Z = {sym: idx for idx, sym in enumerate(_PERIODIC_SYMBOLS) if sym}


def _symbol_to_z(token: str) -> int:
    clean = re.sub(r"[^A-Za-z]", "", token.strip())
    if not clean:
        if token.strip().isdigit():
            return int(token.strip())
        raise ValueError(f"Cannot parse element symbol {token!r}")
    norm = clean[0].upper() + clean[1:].lower()
    if norm in _SYM_TO_Z:
        return _SYM_TO_Z[norm]
    raise ValueError(f"Unknown element symbol {token!r}")

_TORCHSCRIPT_METADATA_KEYS = (
    "r_max",
    "type_names",
    "n_species",
    "nequip_version",
    "model_input_fields",
    "per_species_rescale_shifts",
    "per_species_rescale_scales",
)


def _decode_extra_files(raw_extra: dict[str, bytes]) -> dict[str, str]:
    decoded: dict[str, str] = {}
    for k, v in raw_extra.items():
        if v:
            try:
                decoded[k] = v.decode("utf-8").strip()
            except Exception:
                continue
    return decoded


def load_nequip_model(
    backbone: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> tuple[nn.Module, dict[str, str]]:
    """Load a compiled TorchScript NequIP/Allegro `.pth` model or Python NequIP checkpoint."""
    spec = str(backbone).strip()
    for prefix in ("nequip:", "allegro:"):
        if spec.lower().startswith(prefix):
            spec = spec[len(prefix):].strip()
            break
    spec_path = Path(spec)
    if not spec_path.exists():
        raise FileNotFoundError(
            f"NequIP/Allegro checkpoint {spec_path} does not exist. "
            "Export a deployed model via `nequip-deploy build --train-dir ... model.pth` "
            "or pass an in-memory `nequip_model`."
        )

    # 1) Try native `nequip` loader when installed
    try:
        from nequip.scripts.deploy import load_deployed_model  # type: ignore
        model, metadata = load_deployed_model(spec_path, device=device, freeze=False)
        meta_str = {str(k): str(v) for k, v in (metadata or {}).items()}
        return model, meta_str
    except Exception:
        pass

    # 2) Pure-PyTorch TorchScript loader with `_extra_files` metadata extraction (zero `nequip` dep)
    extra_files = {k: b"" for k in _TORCHSCRIPT_METADATA_KEYS}
    try:
        script_model = torch.jit.load(str(spec_path), map_location=device, _extra_files=extra_files)
        return script_model, _decode_extra_files(extra_files)
    except RuntimeError:
        pass

    # 3) Standard `torch.load` Python nn.Module checkpoint
    loaded = torch.load(str(spec_path), map_location=device, weights_only=False)
    if isinstance(loaded, dict) and "model" in loaded and isinstance(loaded["model"], nn.Module):
        meta = {str(k): str(v) for k, v in loaded.get("metadata", {}).items()}
        return loaded["model"], meta
    if isinstance(loaded, nn.Module):
        return loaded, {}
    raise ValueError(f"Loaded object from {spec_path} is neither a TorchScript nor a PyTorch NequIP module.")


def _resolve_supported_atomic_numbers(
    nequip_model: nn.Module,
    metadata: dict[str, str],
    explicit_atomic_numbers: Optional[Sequence[int]] = None,
) -> list[int]:
    if explicit_atomic_numbers is not None:
        return [int(z) for z in explicit_atomic_numbers]

    if hasattr(nequip_model, "atomic_numbers"):
        return [int(z) for z in torch.as_tensor(getattr(nequip_model, "atomic_numbers")).tolist()]

    type_names: list[str] = []
    if hasattr(nequip_model, "type_names"):
        raw_tn = getattr(nequip_model, "type_names")
        if isinstance(raw_tn, str):
            type_names = raw_tn.split()
        else:
            type_names = [str(x) for x in raw_tn]
    elif "type_names" in metadata and metadata["type_names"]:
        raw_s = metadata["type_names"]
        if raw_s.startswith("["):
            type_names = [str(x) for x in json.loads(raw_s)]
        else:
            type_names = raw_s.split()

    if type_names:
        return [_symbol_to_z(sym) for sym in type_names]

    if "n_species" in metadata and metadata["n_species"].isdigit():
        n_sp = int(metadata["n_species"])
        return list(range(1, n_sp + 1))

    raise ValueError(
        "Could not infer species mapping from `nequip_model` (expected `.atomic_numbers`, `.type_names`, "
        "TorchScript `type_names` metadata, or explicit `atomic_numbers=[...]`)."
    )


def infer_nequip_node_irreps(
    nequip_model: nn.Module,
    explicit_node_irreps: Optional[str] = None,
) -> tuple[o3.Irreps, str]:
    """Return `(node_irreps, feature_key)` for `nequip_model`.

    Checks `explicit_node_irreps`, `.node_features_irreps`, `.node_feats_irreps`, and NequIP's
    `GraphModuleMixin.irreps_out` dictionary. For stripped `nequip-deploy` energy-force models that
    omit intermediate `node_features`, falls back cleanly to per-atom `atomic_energy` (`1x0e`).
    """
    if explicit_node_irreps is not None:
        return o3.Irreps(explicit_node_irreps), "node_features"

    for attr in ("node_features_irreps", "node_feats_irreps"):
        if hasattr(nequip_model, attr):
            key = "node_features" if attr == "node_features_irreps" else "node_feats"
            return o3.Irreps(getattr(nequip_model, attr)), key

    if hasattr(nequip_model, "irreps_out"):
        ir_out = getattr(nequip_model, "irreps_out")
        if isinstance(ir_out, dict):
            for key in ("node_features", "node_feats", "edge_features"):
                if key in ir_out and key != "edge_features":
                    return o3.Irreps(str(ir_out[key])), key
            if "atomic_energy" in ir_out:
                return o3.Irreps("1x0e"), "atomic_energy"

    # Deployed TorchScript energy/force models always expose `atomic_energy` (M, 1) + `forces` (M, 3)
    return o3.Irreps("1x0e"), "atomic_energy"


class NequIPEncoder(nn.Module):
    """Adapter turning a NequIP or Allegro model into an O(3)-equivariant YLM `Encoder` drop-in.

    Accepts YLM native coordinates (`pos` and `cell` in Bohr), converts internally to the MLIP's
    length unit (default Angstrom), builds exact O(3)-invariant periodic `edge_cell_shift` vectors,
    and converts conservative forces (`eV/Å`) back to atomic units (`Hartree/Bohr`) as `1x1o`.
    """

    def __init__(
        self,
        backbone: Union[str, Path] = "nequip:deployed.pth",
        nequip_model: Optional[nn.Module] = None,
        *,
        atomic_numbers: Optional[Sequence[int]] = None,
        r_max: Optional[float] = None,
        node_irreps: Optional[str] = None,
        freeze_nequip: bool = True,
        include_nequip_forces: bool = True,
        out_irreps: Optional[str] = None,
        n_motion_vectors: int = 0,
        length_unit_angstrom: float = 1.0,
        energy_unit_ev: float = 1.0,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        metadata: dict[str, str] = {}
        if nequip_model is None:
            nequip_model, metadata = load_nequip_model(backbone=backbone, device=device)
        self.backbone_name = str(backbone)
        self.nequip = nequip_model
        self.metadata = metadata
        self.freeze_nequip = bool(freeze_nequip)
        self.include_nequip_forces = bool(include_nequip_forces)
        self.n_motion_vectors = int(n_motion_vectors)
        self.length_unit_angstrom = float(length_unit_angstrom)
        self.energy_unit_ev = float(energy_unit_ev)

        if self.freeze_nequip:
            self.nequip.eval()
            for p in self.nequip.parameters():
                p.requires_grad_(False)

        z_list = _resolve_supported_atomic_numbers(self.nequip, metadata, atomic_numbers)
        self.supported_atomic_numbers = z_list
        self.num_species = len(z_list)

        if r_max is not None:
            r_max_model = float(r_max)
        elif "r_max" in metadata and metadata["r_max"]:
            r_max_model = float(metadata["r_max"])
        else:
            r_max_model = float(torch.as_tensor(getattr(self.nequip, "r_max", 5.0)).item())
        self.r_max_ang = r_max_model * self.length_unit_angstrom
        self.cutoff = self.r_max_ang / BOHR_TO_ANGSTROM

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

        self.node_feats_irreps, self.feature_key = infer_nequip_node_irreps(self.nequip, node_irreps)
        raw_parts = [str(self.node_feats_irreps)]
        if self.include_nequip_forces:
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
        self.hidden_irreps = self.irreps_out

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_nequip:
            self.nequip.eval()
        return self

    def _nequip_dtype(self, fallback: torch.dtype) -> torch.dtype:
        for p in self.nequip.parameters():
            return p.dtype
        for b in self.nequip.buffers():
            if b.is_floating_point():
                return b.dtype
        return fallback

    def _build_nequip_batch(
        self,
        Z: torch.Tensor,
        pos_ang: torch.Tensor,
        mask: torch.Tensor,
        cell_ang: Optional[torch.Tensor],
        pbc: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Convert padded `(B, N, ...)` tensors into NequIP `AtomicDataDict` tensors."""
        device = pos_ang.device
        B, N = Z.shape
        unit_scale = 1.0 / self.length_unit_angstrom

        flat_to_compact = torch.full((B * N,), -1, dtype=torch.long, device=device)
        positions_list: list[torch.Tensor] = []
        z_compact_list: list[torch.Tensor] = []
        batch_list: list[torch.Tensor] = []
        cells_list: list[torch.Tensor] = []
        pbc_list: list[torch.Tensor] = []
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
            positions_list.append((pos_ang[b, idx] * unit_scale).to(dtype))
            z_compact_list.append(Z[b, idx])
            batch_list.append(torch.full((m,), g_id, dtype=torch.long, device=device))
            if cell_ang is not None:
                c_b = cell_ang[b] if cell_ang.ndim == 3 else cell_ang
                cells_list.append((c_b * unit_scale).to(dtype))
                if pbc is not None:
                    p_b = pbc[b] if pbc.ndim == 2 else pbc
                    pbc_list.append(p_b.to(torch.bool))
                else:
                    pbc_list.append((c_b.norm(dim=-1) > 0).to(torch.bool))
            else:
                cells_list.append(torch.zeros((3, 3), dtype=dtype, device=device))
                pbc_list.append(torch.zeros(3, dtype=torch.bool, device=device))
            running += m

        if running == 0:
            return {}, flat_to_compact, torch.empty(0, dtype=torch.long, device=device)

        pos_compact = torch.cat(positions_list, dim=0)
        z_compact = torch.cat(z_compact_list, dim=0).clamp(min=1, max=118)

        inexact = z_compact[~self._z_is_exact[z_compact]]
        self.last_element_warnings = []
        if inexact.numel() > 0:
            uniq = sorted(set(int(v) for v in inexact.tolist()))
            for uz in uniq:
                mapped_z = self.supported_atomic_numbers[int(self._z_to_species[uz].item())]
                msg = f"Z={uz} not in {self.backbone_name} species; mapped to valence analogue Z={mapped_z}."
                self.last_element_warnings.append(msg)
            warnings.warn("; ".join(self.last_element_warnings), UserWarning, stacklevel=2)

        atom_types = self._z_to_species[z_compact]
        batch_vec = torch.cat(batch_list, dim=0)
        cell_stack = torch.stack(cells_list, dim=0)  # (n_graphs, 3, 3)
        pbc_stack = torch.stack(pbc_list, dim=0)     # (n_graphs, 3)

        if cell_ang is None:
            rel = pos_ang[:, :, None, :] - pos_ang[:, None, :, :]
            dist = rel.norm(dim=-1)
            pair = mask[:, :, None] & mask[:, None, :] & (dist < self.r_max_ang)
            pair &= ~torch.eye(N, dtype=torch.bool, device=device)[None]
            b_idx, i_idx, j_idx = pair.nonzero(as_tuple=True)
            dst_c = flat_to_compact[b_idx * N + i_idx]
            src_c = flat_to_compact[b_idx * N + j_idx]
            # NequIP convention: edge_vectors = pos[edge_index[1]] - pos[edge_index[0]] + edge_cell_shift @ cell
            edge_index = torch.stack([src_c, dst_c], dim=0)
            edge_cell_shift = torch.zeros((dst_c.numel(), 3), dtype=dtype, device=device)
        else:
            dst_flat, src_flat, vec_ang, _ = _periodic_edges(pos_ang, mask, cell_ang, pbc, self.r_max_ang)
            dst_c = flat_to_compact[dst_flat]
            src_c = flat_to_compact[src_flat]
            edge_index = torch.stack([src_c, dst_c], dim=0)
            vec_model = vec_ang.to(dtype) * unit_scale
            shift_cart = (vec_model - (pos_compact[dst_c] - pos_compact[src_c])).detach()
            if dst_c.numel() > 0:
                edge_cells = cell_stack[batch_vec[src_c]]  # (E, 3, 3)
                pinv_cells = torch.linalg.pinv(edge_cells.to(torch.float64)).to(dtype)
                frac_shifts = torch.matmul(shift_cart.unsqueeze(-2), pinv_cells).squeeze(-2)
                edge_cell_shift = torch.round(frac_shifts)
            else:
                edge_cell_shift = torch.zeros((0, 3), dtype=dtype, device=device)

        data = {
            "pos": pos_compact,
            "atom_types": atom_types,
            "edge_index": edge_index,
            "edge_cell_shift": edge_cell_shift,
            "cell": cell_stack,
            "pbc": pbc_stack,
            "batch": batch_vec,
        }
        compact_to_flat = (flat_to_compact >= 0).nonzero(as_tuple=False).squeeze(-1)
        order = torch.argsort(flat_to_compact[compact_to_flat])
        compact_to_flat = compact_to_flat[order]
        return data, flat_to_compact, compact_to_flat

    def _invoke_nequip(
        self,
        data: dict[str, torch.Tensor],
        *,
        need_forces: bool,
        need_grad_for_train: bool,
    ) -> dict[str, torch.Tensor]:
        """Run `self.nequip(data)`, handling both batched GraphModules and single-frame TorchScript exports."""
        def _call_once(inp: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            # Single-frame `nequip-deploy` TorchScript models sometimes expect `(3, 3)` cell and `(3,)` pbc
            if isinstance(self.nequip, torch.jit.ScriptModule) and inp["cell"].shape[0] == 1:
                single = dict(inp)
                single["cell"] = inp["cell"][0]
                single["pbc"] = inp["pbc"][0]
                try:
                    return self.nequip(single)
                except RuntimeError:
                    return self.nequip(inp)
            return self.nequip(inp)

        if need_forces:
            with torch.enable_grad():
                if not need_grad_for_train:
                    data["pos"] = data["pos"].detach().requires_grad_(True)
                elif not data["pos"].requires_grad:
                    data["pos"] = data["pos"].requires_grad_(True)
                out = _call_once(data)
                if "forces" not in out:
                    e_key = "total_energy" if "total_energy" in out else "atomic_energy"
                    if e_key not in out:
                        raise KeyError("NequIP module output lacks both 'forces' and 'total_energy'/'atomic_energy'.")
                    grads = torch.autograd.grad(
                        out[e_key].sum(),
                        data["pos"],
                        create_graph=need_grad_for_train,
                        retain_graph=need_grad_for_train,
                    )[0]
                    out = dict(out)
                    out["forces"] = -grads
                return out

        if self.freeze_nequip and not data["pos"].requires_grad:
            with torch.no_grad():
                return _call_once(data)
        return _call_once(data)

    def predict_energy_and_forces(
        self,
        Z: torch.Tensor,
        pos_angstrom: torch.Tensor,
        cell_angstrom: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate total potential energy (eV) and conservative atomic forces (eV/Å) on a single frame `(N, 3)`."""
        z_1 = Z.reshape(1, -1)
        pos_1 = pos_angstrom.reshape(1, -1, 3)
        mask_1 = torch.ones_like(z_1, dtype=torch.bool, device=pos_1.device)
        dtype = self._nequip_dtype(pos_1.dtype)
        cell_1 = None if cell_angstrom is None else cell_angstrom.reshape(1, 3, 3)
        pbc_1 = None if pbc is None else pbc.reshape(1, 3)
        data, _, _ = self._build_nequip_batch(z_1, pos_1.to(dtype), mask_1, cell_1, pbc_1, dtype)
        out = self._invoke_nequip(data, need_forces=True, need_grad_for_train=False)
        e_raw = out["total_energy"].sum() if "total_energy" in out else out["atomic_energy"].sum()
        energy_ev = e_raw.detach() * self.energy_unit_ev
        forces_ev_ang = out["forces"].detach() * (self.energy_unit_ev / self.length_unit_angstrom)
        return energy_ev, forces_ev_ang

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
        nq_dtype = self._nequip_dtype(out_dtype)

        pos_ang = pos.to(nq_dtype) * BOHR_TO_ANGSTROM
        cell_ang = cell.to(nq_dtype) * BOHR_TO_ANGSTROM if cell is not None else None

        data, _, compact_to_flat = self._build_nequip_batch(Z, pos_ang, mask, cell_ang, pbc, nq_dtype)
        if compact_to_flat.numel() == 0:
            return torch.zeros((B, N, self.irreps_out.dim), dtype=out_dtype, device=pos.device)

        need_grad_for_train = torch.is_grad_enabled() and (not self.freeze_nequip or pos.requires_grad)
        nq_out = self._invoke_nequip(
            data,
            need_forces=self.include_nequip_forces,
            need_grad_for_train=need_grad_for_train,
        )

        if self.feature_key in nq_out:
            node_feats = nq_out[self.feature_key]
        elif "node_features" in nq_out:
            node_feats = nq_out["node_features"]
        elif "node_feats" in nq_out:
            node_feats = nq_out["node_feats"]
        elif "atomic_energy" in nq_out:
            node_feats = nq_out["atomic_energy"].reshape(-1, 1)
        else:
            raise KeyError(
                f"NequIP output keys {list(nq_out.keys())} do not include {self.feature_key!r} or 'atomic_energy'."
            )

        if not need_grad_for_train:
            node_feats = node_feats.detach()
        node_feats = node_feats.to(out_dtype)

        feats_parts = [node_feats]
        if self.include_nequip_forces:
            force_scale = self.energy_unit_ev / self.length_unit_angstrom
            forces_ev_ang = nq_out["forces"] * force_scale
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

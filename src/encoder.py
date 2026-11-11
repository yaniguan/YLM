"""A small SE(3)-equivariant encoder for molecules (TFN / NequIP style).

Input:  atomic numbers Z (B, N), positions pos (B, N, 3), atom mask (B, N)
Output: per-atom irreps features (B, N, hidden_irreps.dim)

Each layer:
  message from j to i = channel-wise TensorProduct( x_j , Y(r_ij) ), weights from a small MLP of |r_ij|
  then a Linear mixes channels (NequIP style)
  x_i  <-  Gate( SelfLinear(x_i) + sum_j message_ij )     (+ residual if shapes match)

Only relative vectors r_ij = pos_i - pos_j are used, so moving the molecule does nothing.
Everything is built from e3nn pieces, so rotating the molecule rotates the features.
"""
import math

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate


def bessel_basis(r, cutoff, n):
    """sin(k pi r / c) / r, times a smooth cosine cutoff. r: (E,) -> (E, n)"""
    k = torch.arange(1, n + 1, device=r.device, dtype=r.dtype)
    r = r.unsqueeze(-1)
    basis = torch.sin(k * math.pi * r / cutoff) / r.clamp(min=1e-6)
    smooth = 0.5 * (torch.cos(math.pi * r / cutoff) + 1.0) * (r < cutoff)
    return basis * smooth


class Layer(nn.Module):
    def __init__(self, irreps_in, irreps_out, irreps_sh, n_radial, avg_neighbors):
        super().__init__()
        irreps_out = o3.Irreps(irreps_out)
        scalars = o3.Irreps([(m, ir) for m, ir in irreps_out if ir.l == 0])
        gated = o3.Irreps([(m, ir) for m, ir in irreps_out if ir.l > 0])
        self.gate = Gate(
            scalars, [nn.functional.silu if ir.p == 1 else torch.tanh for _, ir in scalars],
            o3.Irreps(f"{gated.num_irreps}x0e"), [torch.sigmoid],
            gated,
        )
        # Channel-wise tensor product ("uvu", like NequIP): each input channel is combined with
        # Y(r_ij) on its own, weighted by the radial MLP. Then a Linear mixes the channels.
        # (A fully connected product with per-edge weights is ~100x slower.)
        irreps_in = o3.Irreps(irreps_in)
        wanted = {ir for _, ir in self.gate.irreps_in}
        irreps_mid, instructions = [], []
        for i, (mul, ir_in) in enumerate(irreps_in):
            for j, (_, ir_sh) in enumerate(irreps_sh):
                for ir_out in ir_in * ir_sh:
                    if ir_out in wanted:
                        instructions.append((i, j, len(irreps_mid), "uvu", True))
                        irreps_mid.append((mul, ir_out))
        irreps_mid = o3.Irreps(irreps_mid)
        self.tp = o3.TensorProduct(irreps_in, irreps_sh, irreps_mid, instructions,
                                   internal_weights=False, shared_weights=False)
        self.after_tp = o3.Linear(irreps_mid, self.gate.irreps_in)
        self.radial = FullyConnectedNet([n_radial, 64, self.tp.weight_numel], nn.functional.silu)
        self.self_linear = o3.Linear(irreps_in, self.gate.irreps_in)
        self.avg_neighbors = avg_neighbors
        self.residual = o3.Irreps(irreps_in) == self.gate.irreps_out

    def forward(self, x, src, dst, sh, radial):
        msg = self.tp(x[src], sh, self.radial(radial))
        agg = torch.zeros(x.shape[0], msg.shape[1], device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, msg)
        agg = self.after_tp(agg / math.sqrt(self.avg_neighbors))
        out = self.gate(self.self_linear(x) + agg)
        return x + out if self.residual else out


def _periodic_edges(pos, mask, cell, pbc, cutoff):
    """Build neighbor edges (dst, src, vec, dist) under 1D/2D/3D periodic boundary conditions.

    cell: (B, 3, 3) or (3, 3) lattice vectors as rows in bohr.
    pbc:  optional (B, 3) or (3,) bool mask per lattice vector; defaults to non-zero rows of cell.
    Self-images (i == j) at non-zero lattice shifts S != 0 are included when within cutoff;
    only the trivial zero-shift self-pair (i == j, S == 0) is excluded.
    """
    B, N, _ = pos.shape
    if cell.ndim == 2:
        cell = cell.unsqueeze(0).expand(B, 3, 3)
    if cell.shape != (B, 3, 3):
        raise ValueError(f"Expected cell shape {(B, 3, 3)}, got {tuple(cell.shape)}")
    if pbc is None:
        pbc_mask = cell.norm(dim=-1) > 0
    else:
        pbc_mask = torch.as_tensor(pbc, dtype=torch.bool, device=pos.device)
        if pbc_mask.ndim == 1:
            pbc_mask = pbc_mask.unsqueeze(0).expand(B, 3)
        if pbc_mask.shape != (B, 3):
            raise ValueError(f"Expected pbc shape {(B, 3)}, got {tuple(pbc_mask.shape)}")

    dst_list, src_list, vec_list, dist_list = [], [], [], []
    for b in range(B):
        idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
        M = idx.numel()
        if M == 0:
            continue
        pos_b = pos[b, idx]
        pbc_b = pbc_mask[b]
        if not pbc_b.any():
            rel_b = pos_b[:, None, :] - pos_b[None, :, :]
            dist_b = rel_b.norm(dim=-1)
            keep = (dist_b < cutoff) & ~torch.eye(M, dtype=torch.bool, device=pos.device)
            ii, jj = keep.nonzero(as_tuple=True)
            dst_list.append(b * N + idx[ii])
            src_list.append(b * N + idx[jj])
            vec_list.append(rel_b[ii, jj])
            dist_list.append(dist_b[ii, jj])
            continue

        basis = cell[b, pbc_b]                                        # (n_per, 3)
        if (torch.linalg.svdvals(basis) <= 1e-8).any():
            raise ValueError("Singular periodic cell vectors")
        pinv = torch.linalg.pinv(basis)                               # (3, n_per)
        pos_wrapped = pos_b - torch.floor((pos_b @ pinv).detach()) @ basis

        max_shifts = (torch.ceil(cutoff * pinv.norm(dim=0)).long() + 1).tolist()
        grids = [torch.arange(-m, m + 1, device=pos.device, dtype=torch.long) for m in max_shifts]
        shifts = torch.cartesian_prod(*grids)
        if shifts.ndim == 1:
            shifts = shifts.unsqueeze(-1)
        zero_shift = (shifts == 0).all(dim=-1)
        shift_vecs = shifts.to(pos.dtype) @ basis                     # (K, 3)
        eye = torch.eye(M, dtype=torch.bool, device=pos.device)[:, :, None]

        for start in range(0, shift_vecs.shape[0], 64):
            sv = shift_vecs[start:start + 64]
            zs = zero_shift[start:start + 64]
            rel_b = pos_wrapped[:, None, None, :] - (pos_wrapped[None, :, None, :] + sv[None, None, :, :])
            dist_b = rel_b.norm(dim=-1)
            keep = (dist_b < cutoff) & ~(eye & zs[None, None, :])
            ii, jj, kk = keep.nonzero(as_tuple=True)
            if ii.numel():
                dst_list.append(b * N + idx[ii])
                src_list.append(b * N + idx[jj])
                vec_list.append(rel_b[ii, jj, kk])
                dist_list.append(dist_b[ii, jj, kk])

    if not dst_list:
        empty_i = torch.empty(0, dtype=torch.long, device=pos.device)
        return empty_i, empty_i, torch.empty(0, 3, dtype=pos.dtype, device=pos.device), torch.empty(0, dtype=pos.dtype, device=pos.device)
    return torch.cat(dst_list), torch.cat(src_list), torch.cat(vec_list), torch.cat(dist_list)


class Encoder(nn.Module):
    def __init__(self, hidden_irreps="64x0e + 8x0o + 32x1o + 8x1e + 16x2e",
                 n_layers=3, lmax=2, cutoff=9.45, n_radial=8, n_elements=10, avg_neighbors=15.0):
        super().__init__()
        # cutoff 9.45 bohr = 5 Angstrom
        self.cutoff = cutoff
        self.n_radial = n_radial
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        n_scalar = self.hidden_irreps.count("0e")
        self.embed = nn.Embedding(n_elements, n_scalar)
        layers = []
        irreps_in = o3.Irreps(f"{n_scalar}x0e")
        for _ in range(n_layers):
            layer = Layer(irreps_in, self.hidden_irreps, self.irreps_sh, n_radial, avg_neighbors)
            layers.append(layer)
            irreps_in = layer.gate.irreps_out
        self.layers = nn.ModuleList(layers)
        self.irreps_out = irreps_in

    def forward(self, Z, pos, mask, extra_scalars=None, cell=None, pbc=None):
        """extra_scalars: optional (B, N, n_scalar) invariant features added to the l=0 input embedding
        (e.g. species role / mole fraction for mixtures). None keeps the original behaviour.
        cell: optional (B, 3, 3) periodic lattice vectors as rows in bohr; pbc: optional (B, 3) bool."""
        B, N = Z.shape
        if cell is None:
            # all pairs (i, j) inside the same molecule, both real atoms, i != j, closer than cutoff
            rel = pos[:, :, None, :] - pos[:, None, :, :]            # (B, N, N, 3): pos_i - pos_j
            dist = rel.norm(dim=-1)
            pair = mask[:, :, None] & mask[:, None, :] & (dist < self.cutoff)
            pair &= ~torch.eye(N, dtype=torch.bool, device=Z.device)[None]
            b, i, j = pair.nonzero(as_tuple=True)
            dst = b * N + i                                           # receiver, flat index
            src = b * N + j                                           # sender, flat index
            vec = rel[b, i, j]                                        # r_i - r_j
            edge_dist = dist[b, i, j]
        else:
            dst, src, vec, edge_dist = _periodic_edges(pos, mask, cell, pbc, self.cutoff)
        sh = o3.spherical_harmonics(self.irreps_sh, vec, normalize=True, normalization="component")
        radial = bessel_basis(edge_dist, self.cutoff, self.n_radial)

        x = self.embed(Z.reshape(-1))                             # (B*N, n_scalar)
        if extra_scalars is not None:
            x = x + extra_scalars.reshape(B * N, -1).to(x.dtype)
        for layer in self.layers:
            x = layer(x, src, dst, sh, radial)
        x = x.reshape(B, N, -1) * mask[..., None]
        return x

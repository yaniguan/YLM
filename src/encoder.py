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

    def forward(self, Z, pos, mask):
        B, N = Z.shape
        # all pairs (i, j) inside the same molecule, both real atoms, i != j, closer than cutoff
        rel = pos[:, :, None, :] - pos[:, None, :, :]            # (B, N, N, 3): pos_i - pos_j
        dist = rel.norm(dim=-1)
        pair = mask[:, :, None] & mask[:, None, :] & (dist < self.cutoff)
        pair &= ~torch.eye(N, dtype=torch.bool, device=Z.device)[None]
        b, i, j = pair.nonzero(as_tuple=True)
        dst = b * N + i                                           # receiver, flat index
        src = b * N + j                                           # sender, flat index
        vec = rel[b, i, j]                                        # r_i - r_j
        sh = o3.spherical_harmonics(self.irreps_sh, vec, normalize=True, normalization="component")
        radial = bessel_basis(dist[b, i, j], self.cutoff, self.n_radial)

        x = self.embed(Z.reshape(-1))                             # (B*N, n_scalar)
        for layer in self.layers:
            x = layer(x, src, dst, sh, radial)
        x = x.reshape(B, N, -1) * mask[..., None]
        return x

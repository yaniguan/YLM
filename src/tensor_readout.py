"""Opt-in symmetric tensor head (0e + 2e).

The default YLM head is still one polar vector. This head is used only when
YLM(..., tensor_readout=True). It returns one symmetric 3x3 matrix.

Quadrupole files from scripts/12_make_quadrupole.py are dicts of traceless
symmetric 3x3 matrices. The trace is the 0e part, so those labels have 0e = 0.
The 2e part is the shape. A polarizability tensor can use the same head;
its 0e part is the isotropic average and is not zero.
"""
import pickle

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.io import CartesianTensor


_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


class TensorReadout(nn.Module):
    """Map pooled irreps to one symmetric 3x3 matrix."""

    def __init__(self, irreps_in):
        super().__init__()
        cartesian = CartesianTensor("ij=ji")
        self.linear = o3.Linear(irreps_in, cartesian)
        rtp = cartesian.reduced_tensor_products()
        # Fixed basis. Not a trained weight, and not part of the state dict.
        basis = rtp.change_of_basis.detach().clone()
        self.register_buffer("change_of_basis", basis, persistent=False)

    def forward(self, features):
        """features: (..., irreps) -> (..., 3, 3), symmetric."""
        irreps = self.linear(features)
        basis = self.change_of_basis.to(dtype=irreps.dtype, device=irreps.device)
        flat = irreps @ basis.flatten(start_dim=-2)
        return flat.reshape(*irreps.shape[:-1], 3, 3)


def unique_entries(matrix):
    """Six independent entries of a symmetric 3x3 matrix. (..., 3, 3) -> (..., 6)."""
    parts = [matrix[..., i, j] for i, j in _PAIRS]
    return torch.stack(parts, dim=-1)


def quadrupole_loss(pred, target):
    """Mean square error on the six independent entries."""
    return (unique_entries(pred) - unique_entries(target)).pow(2).mean()


def check_quadrupole_matrix(q):
    """Return one traceless symmetric 3x3 matrix as a float64 tensor.

    This is the array stored for each molecule in data/qm9_quad.pkl.
    """
    matrix = torch.as_tensor(q, dtype=torch.float64)
    if matrix.shape != (3, 3):
        raise ValueError("quadrupole matrix must have shape (3, 3)")
    if not torch.allclose(matrix, matrix.T, atol=1e-5, rtol=1e-5):
        raise ValueError("quadrupole matrix must be symmetric")
    if matrix.trace().abs() > 1e-4:
        raise ValueError("quadrupole matrix must be traceless")
    return matrix


def load_quadrupole_labels(path):
    """Load data/qm9_quad.pkl: molecule index -> (3, 3) matrix."""
    with open(path, "rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict) or len(raw) == 0:
        raise ValueError("quadrupole file must be a non-empty dict of molecule index -> 3x3 matrix")
    return {int(k): check_quadrupole_matrix(v) for k, v in raw.items()}


def rotate_matrix(q, R):
    """Rotate a 3x3 matrix, or a batch of them, by R q R^T."""
    rotation = R.to(dtype=q.dtype, device=q.device)
    return rotation @ q @ rotation.transpose(-1, -2)


def tensor_equivariance_error(y, y0, R):
    """Relative error ||y - R y0 R^T|| / ||y0|| for each batch row.

    y and y0 have shape (B, 3, 3). R has shape (3, 3).
    """
    rotated = rotate_matrix(y0, R)
    num = (y - rotated).reshape(y.shape[0], -1).norm(dim=-1)
    den = y0.reshape(y0.shape[0], -1).norm(dim=-1).clamp(min=1e-12)
    return num / den


def build_arrays(geom_labels, quad, n_max=29):
    """Stack geometries and quadrupole matrices for the molecules that have both.

    geom_labels is the list in data/qm9_xtb.pkl. quad is the dict from
    load_quadrupole_labels. Row order follows the sorted molecule index.
    """
    ids = sorted(int(k) for k in quad)
    if not ids:
        raise ValueError("quadrupole dict is empty")
    pos_dtype = torch.float32
    Z = torch.zeros(len(ids), n_max, dtype=torch.long)
    pos = torch.zeros(len(ids), n_max, 3, dtype=pos_dtype)
    mask = torch.zeros(len(ids), n_max, dtype=torch.bool)
    target = torch.zeros(len(ids), 3, 3, dtype=pos_dtype)
    for row, mol in enumerate(ids):
        if mol < 0 or mol >= len(geom_labels):
            raise ValueError(f"quadrupole index {mol} is outside the geometry list")
        lab = geom_labels[mol]
        numbers = lab["numbers"]
        xyz = lab["pos"]
        n_atoms = len(numbers)
        if n_atoms > n_max or len(xyz) != n_atoms:
            raise ValueError(f"molecule {mol} does not fit in the padded geometry")
        Z[row, :n_atoms] = torch.as_tensor(numbers, dtype=torch.long)
        pos[row, :n_atoms] = torch.as_tensor(xyz, dtype=pos_dtype)
        mask[row, :n_atoms] = True
        target[row] = check_quadrupole_matrix(quad[mol]).to(dtype=pos_dtype)
    return dict(mol=torch.tensor(ids, dtype=torch.long), Z=Z, pos=pos, mask=mask, target=target)


def split_rows(mol_ids):
    """Hold out molecules by index: 8 mod 10 is val, 9 mod 10 is test."""
    rows = {"train": [], "val": [], "test": []}
    for row, mol in enumerate(mol_ids.tolist()):
        if mol % 10 == 8:
            rows["val"].append(row)
        elif mol % 10 == 9:
            rows["test"].append(row)
        else:
            rows["train"].append(row)
    return {name: torch.tensor(index, dtype=torch.long) for name, index in rows.items()}


def make_batch(arrays, rows, device, R=None):
    """Inputs for YLM in taskid mode, plus the (possibly rotated) 3x3 target."""
    pos = arrays["pos"][rows]
    target = arrays["target"][rows]
    if R is not None:
        rotation = R.to(dtype=pos.dtype, device=pos.device)
        pos = pos @ rotation.T
        target = rotate_matrix(target, rotation)
    count = len(rows)
    text = torch.zeros(count, 1, 1, dtype=pos.dtype, device=device)
    text_mask = torch.ones(count, 1, dtype=torch.bool, device=device)
    task = torch.zeros(count, dtype=torch.long, device=device)
    inputs = (
        arrays["Z"][rows].to(device),
        pos.to(device),
        arrays["mask"][rows].to(device),
        text,
        text_mask,
        task,
    )
    return inputs, target.to(device)

"""Controlled baselines for TensorQA-v0 (see 04_data_and_benchmark.md).

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

EquiLLMBaseline  "invariants into Transformer, equivariant vector adapter after"
    Same equivariant encoder, but the language Transformer only attends over text tokens and
    per-atom O(3) invariants. Equivariant l=1 atom vectors bypass the language module and are
    scaled afterwards by invariant per-atom/global gates before a final o3.Linear readout.

FrameAveragingBaseline  "PCA O(3) frame averaging around a plain Transformer"
    No e3nn irreps. Projects centered coordinates onto all 8 sign choices of the covariance
    eigenvectors, runs a plain coordinate+text Transformer in each canonical frame, and averages
    the rotated 3D vector predictions back in the lab frame. Exact O(3) equivariance holds when
    the three principal eigenvalues are distinct.
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


class EquiLLMBaseline(nn.Module):
    def __init__(self, text_dim, hidden_irreps="64x0e + 8x0o + 32x1o + 8x1e + 16x2e",
                 n_layers=3, d=128, n_vec=32, n_bridge_layers=2, n_heads=4, text_mode="text", n_tasks=2):
        super().__init__()
        self.text_mode = text_mode
        self.n_vec = n_vec
        self.encoder = Encoder(hidden_irreps=hidden_irreps, n_layers=n_layers)
        n_inv = self.encoder.irreps_out.num_irreps
        self.task_embed = nn.Embedding(n_tasks, text_dim)
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d), nn.SiLU())
        self.atom_proj = nn.Sequential(nn.LayerNorm(n_inv), nn.Linear(n_inv, d), nn.SiLU())
        self.type_embed = nn.Embedding(2, d)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=2 * d, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.bridge = nn.TransformerEncoder(enc_layer, num_layers=n_bridge_layers, enable_nested_tensor=False)
        self.token_weight = nn.Linear(d, 1)
        self.atom_score = nn.Linear(d, 1)
        self.atom_gate = nn.Linear(d, n_vec)
        self.gate_pick = nn.Linear(d, n_vec)
        self.gate_sum = nn.Linear(d, n_vec)
        self.vec_bypass = o3.Linear(self.encoder.irreps_out, f"{n_vec}x1o")
        self.readout = o3.Linear(f"{n_vec}x1o", "1x1o")

    def forward(self, Z, pos, atom_mask, text, text_mask, task, atom=None):
        if self.text_mode == "taskid":
            text = self.task_embed(task)[:, None, :]
            text_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
        B, T, _ = text.shape
        N = Z.shape[1]
        h_text = self.text_proj(text) + self.type_embed.weight[0]
        x = self.encoder(Z, pos, atom_mask)
        h_atom = self.atom_proj(invariants(x, self.encoder.irreps_out)) + self.type_embed.weight[1]

        seq = torch.cat([h_text, h_atom], dim=1)
        valid = torch.cat([text_mask, atom_mask], dim=1)
        out = self.bridge(seq, src_key_padding_mask=~valid)
        z_text, z_atom = out[:, :T], out[:, T:]

        tw = self.token_weight(z_text).squeeze(-1).masked_fill(~text_mask, float("-inf")).softmax(-1)
        z_global = torch.einsum("bt,btd->bd", tw, z_text)

        attn = self.atom_score(z_atom).squeeze(-1).masked_fill(~atom_mask, float("-inf")).softmax(-1)
        a_gate = self.atom_gate(z_atom)
        v = (self.vec_bypass(x) * atom_mask[..., None]).reshape(B, N, self.n_vec, 3)

        picked = torch.einsum("bn,bnc,bncm->bcm", attn, a_gate, v)
        total = v.sum(dim=1)
        g_pick = torch.sigmoid(self.gate_pick(z_global))[..., None]
        g_sum = self.gate_sum(z_global)[..., None]
        fused = (g_pick * picked + g_sum * total).reshape(B, self.n_vec * 3)
        return self.readout(fused), attn


class FrameAveragingBaseline(nn.Module):
    def __init__(self, text_dim, d=128, n_atom_layers=2, n_heads=4, n_elements=10,
                 text_mode="text", n_tasks=2):
        super().__init__()
        self.text_mode = text_mode
        self.d = d
        self.task_embed = nn.Embedding(n_tasks, text_dim)
        self.atom_embed = nn.Embedding(n_elements, d)
        self.pos_proj = nn.Linear(3, d)
        atom_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=2 * d, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.atom_encoder = nn.TransformerEncoder(atom_layer, num_layers=n_atom_layers, enable_nested_tensor=False)
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d), nn.SiLU())
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.token_weight = nn.Linear(d, 1)
        self.gate_pick = nn.Linear(d, d)
        self.gate_sum = nn.Linear(d, d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 3))
        signs = torch.tensor([[sx, sy, sz] for sx in (-1.0, 1.0)
                              for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
        self.register_buffer("signs", signs, persistent=False)

    def _pca_frames(self, centered):
        with torch.autocast(device_type=centered.device.type, enabled=False):
            c64 = centered.double()
            cov = torch.einsum("bni,bnj->bij", c64, c64)
            _, eigvecs = torch.linalg.eigh(cov)
            frames = eigvecs[:, None, :, :] * self.signs.to(device=c64.device, dtype=c64.dtype)[None, :, None, :]
        return frames.to(centered.dtype)

    def forward(self, Z, pos, atom_mask, text, text_mask, task, atom=None):
        if self.text_mode == "taskid":
            text = self.task_embed(task)[:, None, :]
            text_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=text.device)
        B, N = Z.shape
        F = self.signs.shape[0]
        count = atom_mask.sum(dim=1, keepdim=True).clamp(min=1).to(pos.dtype)
        mean_pos = (pos * atom_mask[..., None]).sum(dim=1, keepdim=True) / count[..., None]
        centered = (pos - mean_pos) * atom_mask[..., None]
        frames = self._pca_frames(centered)
        canon_pos = torch.einsum("bni,bfij->bfnj", centered, frames) * atom_mask[:, None, :, None]

        z_emb = self.atom_embed(Z)[:, None, :, :].expand(B, F, N, self.d)
        atom_in = (z_emb + self.pos_proj(canon_pos)).reshape(B * F, N, self.d)
        atom_pad = (~atom_mask)[:, None, :].expand(B, F, N).reshape(B * F, N)
        atom_states = self.atom_encoder(atom_in, src_key_padding_mask=atom_pad).reshape(B, F, N, self.d)
        atom_states = atom_states * atom_mask[:, None, :, None]

        h = self.text_proj(text)
        q = self.q(h)
        k = self.k(atom_states)
        v = self.v(atom_states) * atom_mask[:, None, :, None]
        scores = torch.einsum("btd,bfnd->bftn", q, k) / math.sqrt(self.d)
        scores = scores.masked_fill(~atom_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)

        picked = torch.einsum("bftn,bfnd->bftd", attn, v)
        total = v.sum(dim=2, keepdim=True)
        g_pick = torch.sigmoid(self.gate_pick(h))[:, None, :, :]
        g_sum = self.gate_sum(h)[:, None, :, :]
        tok_out = g_pick * picked + g_sum * total

        tw = self.token_weight(h).squeeze(-1).masked_fill(~text_mask, float("-inf")).softmax(-1)
        pooled = torch.einsum("bt,bftd->bfd", tw, tok_out)
        y_canon = self.head(pooled)
        y = torch.einsum("bfij,bfj->bi", frames, y_canon) / float(F)
        attn_pool = torch.einsum("bt,btn->bn", tw, attn.mean(dim=1))
        return y, attn_pool

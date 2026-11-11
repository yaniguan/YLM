"""YLM language model: a decoder LM that reads molecular geometry through irreps tokens.

Arms of the Oct-30 ablation (docs/EXPERIMENT_DESIGN_oct30.md):
  text     no geometry pathway (the SMILES / SMILES+xyz text arms). No extra parameters.
  prefix   per-atom invariant features of the equivariant encoder -> soft tokens prepended to the sequence.
  xattnA/B/C
           an IrrepsAdapter after decoder layer k:
               e_t = IrrepsToText(LN(h_t), atom_irreps)          (repo option A/B/C, queries = LM states)
               h_t <- h_t + tanh(gate) * Linear(LN(invariants(e_t)))
           gate starts at 0, so at initialisation the model is exactly the text-only LM.

Geometry is only seen through O(3)-invariant quantities (even scalars, |odd scalars|, channel norms),
so the LM loss is invariant to rotations, reflections and translations of the input coordinates
(tests/test_ylm_lm.py checks this).

Mixtures (electrolytes): atoms of all species are concatenated into one "molecule"; species are placed
more than one cutoff apart (src/tasks/conformers.build_mixture), so the encoder builds no edge between
species. Every atom gets two invariant extras added to its l=0 input channel:
    role_embed(role)  +  amount_embed(amount)
role = molecule / salt / solvent, amount = recorded ratio fraction (solvents) or concentration (salt).
The prompt retains ratio basis and concentration units; no mass/volume-to-mole conversion is assumed.
We deliberately do NOT embed the arbitrary species index (0, 1, 2, ...): that would make the
representation depend on the order species are listed in, which carries no chemistry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from e3nn import o3

from src.encoder import Encoder
from src.irreps_to_text import IrrepsToText, channel_index, invariants

MODES = ("text", "prefix", "xattnA", "xattnB", "xattnC")
ROLES = {"molecule": 0, "salt": 1, "solvent": 2, "other": 3}

ENCODER_PRESETS = {
    "tiny": dict(hidden_irreps="16x0e + 4x0o + 8x1o + 4x1e + 4x2e", n_layers=2),
    "repo_tiny": dict(hidden_irreps="8x0e + 2x0o + 4x1o + 2x1e + 2x2e", n_layers=3),   # 04_train.py --tiny
    "small": dict(hidden_irreps="64x0e + 8x0o + 32x1o + 8x1e + 16x2e", n_layers=3),    # repo default
}
# Option C runs a per-token, text-weighted CG self product: its cost grows with the cube of the
# multiplicities, so it gets smaller cross irreps (parameter counts are reported for every arm).
CROSS_PRESETS = {
    "tiny": {"A": "8x0e + 8x1o + 4x1e + 4x2e", "B": "8x0e + 8x1o + 4x1e + 4x2e",
             "C": "4x0e + 4x1o + 2x1e + 2x2e"},
    "repo_tiny": {"A": "4x0e + 4x1o + 2x1e + 2x2e", "B": "4x0e + 4x1o + 2x1e + 2x2e",
                  "C": "4x0e + 4x1o + 2x1e + 2x2e"},
    "small": {"A": "32x0e + 32x1o + 8x1e + 8x2e", "B": "32x0e + 32x1o + 8x1e + 8x2e",
              "C": "8x0e + 8x1o + 4x1e + 4x2e"},
}
LM_PRESETS = {
    "tiny": dict(n_layer=4, n_embd=128, n_head=4),
    "small": dict(n_layer=12, n_embd=768, n_head=12),     # GPT-2 small shape
}


@dataclass
class YLMConfig:
    mode: str = "text"
    lm: str = "tiny"                       # tiny | small | hf:<name>
    vocab_size: int = 0                    # from-scratch GPT-2 only
    n_positions: int = 1024
    n_layer: Optional[int] = None          # override the preset
    n_embd: Optional[int] = None
    n_head: Optional[int] = None
    bos_token_id: int = 1
    eos_token_id: int = 2
    encoder: str = "tiny"
    n_elements: int = 100
    cutoff: float = 9.45                   # bohr
    adapter_layers: Optional[list] = None  # default: [n_layer // 2]
    cross_irreps: Optional[str] = None     # default: CROSS_PRESETS[encoder][option]
    b_score_mode: str = "legacy"
    key_dim: int = 64
    vec_readout: bool = False
    vec_token_id: Optional[int] = None
    vec_loss_weight: float = 1.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "YLMConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def collate_geometry(geoms, device=None, dtype=None) -> dict:
    """List of dicts {Z (N,), pos (N,3) bohr, role (N,), amount (N,)} -> padded tensors."""
    dtype = dtype or torch.get_default_dtype()
    B = len(geoms)
    N = max(1, max((len(g["Z"]) for g in geoms if g is not None), default=0))
    Z = torch.zeros(B, N, dtype=torch.long)
    pos = torch.zeros(B, N, 3, dtype=dtype)
    mask = torch.zeros(B, N, dtype=torch.bool)
    role = torch.zeros(B, N, dtype=torch.long)
    amount = torch.zeros(B, N, dtype=dtype)
    for b, g in enumerate(geoms):
        if g is None:
            continue
        n = len(g["Z"])
        if n == 0:
            continue
        Z[b, :n] = torch.as_tensor(np.asarray(g["Z"]), dtype=torch.long)
        pos[b, :n] = torch.as_tensor(np.asarray(g["pos"]), dtype=dtype)
        mask[b, :n] = True
        if g.get("role") is not None:
            role[b, :n] = torch.as_tensor(np.asarray(g["role"]), dtype=torch.long)
        amount[b, :n] = torch.as_tensor(np.asarray(g["amount"]), dtype=dtype) if g.get("amount") is not None else 1.0
    out = dict(Z=Z, pos=pos, atom_mask=mask, role=role, amount=amount)
    return {k: v.to(device) for k, v in out.items()} if device is not None else out


class IrrepsTokenizer(nn.Module):
    """Equivariant encoder (n_elements=100) -> per-atom irreps ("irreps tokens") with mixture extras."""

    def __init__(self, preset: str = "tiny", n_elements: int = 100, cutoff: float = 9.45, **overrides):
        super().__init__()
        kw = dict(ENCODER_PRESETS[preset])
        kw.update(overrides)
        self.encoder = Encoder(n_elements=n_elements, cutoff=cutoff, **kw)
        n_scalar = self.encoder.hidden_irreps.count("0e")
        self.role_embed = nn.Embedding(len(ROLES), n_scalar)
        self.amount_embed = nn.Linear(1, n_scalar, bias=False)
        nn.init.zeros_(self.role_embed.weight)    # start as the plain encoder
        nn.init.zeros_(self.amount_embed.weight)
        self.irreps_out = self.encoder.irreps_out

    @property
    def n_invariants(self) -> int:
        return self.irreps_out.num_irreps

    def forward(self, Z, pos, atom_mask, role=None, amount=None):
        dtype = self.role_embed.weight.dtype
        with torch.autocast(device_type=pos.device.type, enabled=False):
            pos = pos.to(dtype)
            extra = self.role_embed(role if role is not None else torch.zeros_like(Z))
            if amount is not None:
                extra = extra + self.amount_embed(amount.to(dtype)[..., None])
            return self.encoder(Z, pos, atom_mask, extra_scalars=extra)

    def invariant_tokens(self, x):
        """Parity-correct invariants per atom (even scalars, |odd scalars|, channel norms)."""
        return invariants(x, self.irreps_out)


class IrrepsAdapter(nn.Module):
    """h <- h + tanh(gate) * Linear(LN(invariants(IrrepsToText(LN(h), atom_irreps))))."""

    def __init__(self, lm_dim: int, atom_irreps, option: str = "A", cross_irreps: Optional[str] = None,
                 key_dim: int = 64, b_score_mode: str = "legacy"):
        super().__init__()
        cross_irreps = cross_irreps or CROSS_PRESETS["tiny"][option]
        self.norm_h = nn.LayerNorm(lm_dim)
        self.cross = IrrepsToText(lm_dim, atom_irreps, cross_irreps, key_dim=key_dim, option=option, b_score_mode=b_score_mode)
        n_inv = self.cross.out_irreps.num_irreps
        self.norm_e = nn.LayerNorm(n_inv)
        self.proj = nn.Linear(n_inv, lm_dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, h, x, atom_mask, has_atoms=None, return_side: bool = False):
        dtype = self.proj.weight.dtype
        with torch.autocast(device_type=h.device.type, enabled=False):
            e, _ = self.cross(self.norm_h(h.to(dtype)), x.to(dtype), atom_mask)
            if has_atoms is not None:
                e = e * has_atoms[:, None, None].to(dtype)
            delta = torch.tanh(self.gate) * self.proj(self.norm_e(invariants(e, self.cross.out_irreps)))
        out = h + delta.to(h.dtype)
        return (out, e) if return_side else out


class _LayerHook:
    """Forward hook on a decoder layer. A class (not a closure) so deepcopy re-targets it correctly."""

    def __init__(self, owner: "YLMCausalLM", index: int):
        self.owner = owner
        self.index = index

    def __call__(self, module, inputs, output):
        ctx = self.owner._ctx
        if ctx is None:
            return output
        adapter = self.owner.adapters[self.index]
        x, mask, has = ctx
        need_side = self.owner.vec_head is not None
        if isinstance(output, tuple):
            res = adapter(output[0], x, mask, has, return_side=need_side)
            if need_side:
                out_h, e = res
                self.owner._vec_stream = e if self.owner._vec_stream is None else self.owner._vec_stream + e
                return (out_h,) + tuple(output[1:])
            return (res,) + tuple(output[1:])
        if torch.is_tensor(output):
            res = adapter(output, x, mask, has, return_side=need_side)
            if need_side:
                out_h, e = res
                self.owner._vec_stream = e if self.owner._vec_stream is None else self.owner._vec_stream + e
                return out_h
            return res
        raise TypeError(f"unsupported decoder layer output type {type(output)}")


def _decoder_layers(lm) -> nn.ModuleList:
    for path in ("transformer.h", "model.layers", "model.decoder.layers", "gpt_neox.layers"):
        obj = lm
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    n = getattr(lm.config, "num_hidden_layers", None)
    for m in lm.modules():
        if isinstance(m, nn.ModuleList) and len(m) == n:
            return m
    raise ValueError("could not find the decoder layers of this LM")


def build_lm(cfg: YLMConfig):
    if cfg.lm.startswith("hf:"):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(cfg.lm[3:]).float()
    from transformers import GPT2Config, GPT2LMHeadModel
    size = dict(LM_PRESETS[cfg.lm])
    for k in ("n_layer", "n_embd", "n_head"):
        if getattr(cfg, k) is not None:
            size[k] = getattr(cfg, k)
    if cfg.vocab_size <= 0:
        raise ValueError("vocab_size is required for a from-scratch GPT-2")
    conf = GPT2Config(vocab_size=cfg.vocab_size, n_positions=cfg.n_positions, bos_token_id=cfg.bos_token_id,
                      eos_token_id=cfg.eos_token_id, **size)
    return GPT2LMHeadModel(conf)


def token_nll(logits, labels):
    """Next-token NLL. logits (B, L, V), labels (B, L) with -100 = ignore -> (B, L-1), 0 where ignored."""
    B, L, V = logits.shape
    nll = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, V).float(), labels[:, 1:].reshape(-1),
                                      ignore_index=-100, reduction="none")
    return nll.view(B, L - 1)


class YLMCausalLM(nn.Module):
    def __init__(self, cfg: YLMConfig):
        super().__init__()
        if cfg.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.cfg = cfg
        self.mode = cfg.mode
        self.lm = build_lm(cfg)
        d = self.lm.config.hidden_size
        self.irreps: Optional[IrrepsTokenizer] = None
        self.prefix: Optional[nn.Module] = None
        self.adapters = nn.ModuleList()
        self.adapter_layers: list = []
        self.vec_norm: Optional[nn.Module] = None
        self.vec_gate: Optional[nn.Module] = None
        self.vec_head: Optional[nn.Module] = None
        self._ctx = None
        self._vec_stream = None
        if cfg.vec_readout and self.mode not in ("xattnA", "xattnB", "xattnC"):
            raise ValueError("vec_readout requires xattnA, xattnB, or xattnC mode")
        if self.mode == "text":
            return
        self.irreps = IrrepsTokenizer(cfg.encoder, n_elements=cfg.n_elements, cutoff=cfg.cutoff)
        if self.mode == "prefix":
            n_inv = self.irreps.n_invariants
            self.prefix = nn.Sequential(nn.LayerNorm(n_inv), nn.Linear(n_inv, d), nn.GELU(), nn.Linear(d, d))
            return
        option = self.mode[-1]
        layers = _decoder_layers(self.lm)
        self.adapter_layers = list(cfg.adapter_layers) if cfg.adapter_layers else [len(layers) // 2]
        cross = cfg.cross_irreps or CROSS_PRESETS[cfg.encoder][option]
        for i, k in enumerate(self.adapter_layers):
            self.adapters.append(IrrepsAdapter(d, self.irreps.irreps_out, option, cross, cfg.key_dim, cfg.b_score_mode))
            layers[k].register_forward_hook(_LayerHook(self, i))
        if cfg.vec_readout:
            out_ir = o3.Irreps(cross)
            self.vec_norm = nn.LayerNorm(d)
            self.vec_gate = nn.Linear(d, out_ir.num_irreps)
            self.vec_head = o3.Linear(out_ir, "1x1o")
            self.register_buffer("_vec_expand", channel_index(out_ir), persistent=False)

    @property
    def needs_geometry(self) -> bool:
        return self.mode != "text"

    def encode_geometry(self, geom: dict):
        """-> (atom irreps x, atom_mask). Reusable across decode steps."""
        x = self.irreps(geom["Z"], geom["pos"], geom["atom_mask"], geom.get("role"), geom.get("amount"))
        return x, geom["atom_mask"]

    def forward(self, input_ids, attention_mask, labels=None, geom=None, geom_feats=None,
                output_hidden: bool = False, vec_targets=None, vec_mask=None) -> dict:
        emb = self.lm.get_input_embeddings()(input_ids)
        mask = attention_mask.long()
        n_prefix = 0
        if self.needs_geometry:
            if geom_feats is None:
                if geom is None:
                    raise ValueError(f"mode {self.mode} needs geometry")
                geom_feats = self.encode_geometry(geom)
            x, atom_mask = geom_feats
            if self.mode == "prefix":
                pre = self.prefix(self.irreps.invariant_tokens(x).to(self.prefix[1].weight.dtype))
                pre = pre * atom_mask[..., None].to(pre.dtype)
                emb = torch.cat([pre.to(emb.dtype), emb], dim=1)
                mask = torch.cat([atom_mask.long(), mask], dim=1)
                n_prefix = pre.shape[1]
            else:
                has = atom_mask.any(1)
                safe = atom_mask.clone()
                safe[~has, 0] = True               # avoid softmax over nothing; output is zeroed below
                self._ctx = (x, safe, has)
        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
        capacity = getattr(self.lm.config, "max_position_embeddings", self.cfg.n_positions)
        if emb.shape[1] > capacity:
            raise ValueError(f"LM context {emb.shape[1]} exceeds capacity {capacity}; reduce max_len/prefix size")
        self._vec_stream = None
        try:
            out = self.lm(inputs_embeds=emb, attention_mask=mask, position_ids=position_ids,
                          output_hidden_states=(output_hidden or self.vec_head is not None), use_cache=False)
            vec_stream = self._vec_stream
        finally:
            self._ctx = None
            self._vec_stream = None
        logits = out.logits[:, n_prefix:]
        res = {"logits": logits, "n_prefix": n_prefix}
        if output_hidden:
            res["hidden"] = out.hidden_states[-1][:, n_prefix:]
        if labels is not None:
            nll = token_nll(logits, labels)
            n = (labels[:, 1:] != -100).sum()
            res["nll"] = nll
            res["loss"] = nll.sum() / n.clamp(min=1)
        if self.vec_head is not None:
            with torch.autocast(device_type=emb.device.type, enabled=False):
                vdtype = self.vec_head.weight.dtype
                h_last = out.hidden_states[-1][:, n_prefix:].to(vdtype)
                g_vec = torch.sigmoid(self.vec_gate(self.vec_norm(h_last)))[..., self._vec_expand]
                vec_all = self.vec_head(g_vec * vec_stream.to(vdtype))
            res["vec_all"] = vec_all
            active = None
            if vec_mask is not None:
                active = vec_mask.bool() & attention_mask.bool()
            elif self.cfg.vec_token_id is not None:
                active = (input_ids == self.cfg.vec_token_id) & attention_mask.bool()
            if active is not None:
                res["vec_mask"] = active
                res["vec"] = vec_all[active]
            if vec_targets is not None:
                if active is None or not bool(active.any()):
                    raise ValueError("vec_targets requires at least one active <vec> position")
                vt = vec_targets.to(device=vec_all.device, dtype=vec_all.dtype)
                if vt.ndim == 3 and vt.shape == vec_all.shape:
                    diff = (vec_all - vt)[active]
                elif vt.ndim == 2 and vt.shape == (vec_all.shape[0], 3) and bool((active.sum(1) == 1).all()):
                    diff = (vec_all - vt[:, None, :])[active]
                elif vt.ndim == 2 and vt.shape == (int(active.sum()), 3):
                    diff = vec_all[active] - vt
                else:
                    raise ValueError("vec_targets shape does not match active <vec> positions")
                vec_loss = diff.square().mean()
                res["vec_loss"] = vec_loss
                res["loss"] = (res["loss"] if "loss" in res else 0.0) + self.cfg.vec_loss_weight * vec_loss
        elif vec_targets is not None:
            raise ValueError("vec_targets requires vec_readout=True")
        return res

    def param_counts(self) -> dict:
        def count(mods):
            seen, total = set(), 0
            for m in mods:
                if m is None:
                    continue
                for p in m.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        total += p.numel()
            return total
        lm = count([self.lm])
        emb = count([self.lm.get_input_embeddings()])
        wpe = getattr(getattr(self.lm, "transformer", None), "wpe", None)
        vec_mods = [self.vec_norm, self.vec_gate, self.vec_head]
        return {
            "lm": lm,
            "lm_non_embedding": lm - emb - count([wpe]),
            "encoder": count([self.irreps]),
            "adapters": count([self.adapters]),
            "prefix_projector": count([self.prefix]),
            "vec_readout": count(vec_mods),
            "geometry_pathway": count([self.irreps, self.adapters, self.prefix] + vec_mods),
            "total": count([self]),
        }

    def gates(self) -> list:
        return [float(torch.tanh(a.gate)) for a in self.adapters]

"""Dual-engine YLMProbe runtime (packaged v0 YLM + YLMCausalLM <vec>) and live O(3) verifier."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Optional, Union

import numpy as np
import torch
from e3nn import o3

import src  # noqa: F401 - register safe globals for torch.load
from src.model import YLM
from src.tasks.common import Example, Seg, collate, decode_with_vectors, load_run, model_inputs
from src.tasks.tokenizer import CharTokenizer, VEC_TOKEN
from src.toolkit.ingest import (
    AU_DIPOLE_TO_DEBYE,
    AU_FORCE_TO_EV_ANG,
    BOHR_TO_ANGSTROM,
    StructureFrame,
    load_structure,
    symbol_for_z,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
V0_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "matscibert": "m3rg-iitd/matscibert",
}

# Valence-group mapping into Z=1..9 for legacy n_elements=10 QM9 v0 checkpoints.
_VALENCE_ANALOGUE_Z = {
    11: 3, 19: 3, 37: 3, 55: 3, 12: 4, 20: 4,
    13: 5, 31: 5,
    14: 6, 32: 6, 50: 6,
    15: 7, 33: 7, 51: 7,
    16: 8, 34: 8, 52: 8,
    17: 9, 35: 9, 53: 9,
}


def _map_elements_for_limit(Z: np.ndarray, n_elements: int) -> tuple[np.ndarray, Optional[str]]:
    if int(Z.max(initial=0)) < n_elements:
        return Z, None
    mapped = Z.copy()
    remapped_pairs = []
    for idx, z in enumerate(Z.tolist()):
        if z >= n_elements:
            target = _VALENCE_ANALOGUE_Z.get(z, 3 if 21 <= z <= 30 or 39 <= z <= 48 else 6)
            target = min(target, n_elements - 1)
            mapped[idx] = target
            remapped_pairs.append(f"{symbol_for_z(z)}(Z={z})->{symbol_for_z(target)}(Z={target})")
    uniq = ", ".join(dict.fromkeys(remapped_pairs))
    warn = (
        f"Checkpoint supports Z < {n_elements} (QM9 elements); remapped heavy atoms via valence analogues: {uniq}. "
        f"Use a YLMCausalLM checkpoint (n_elements=100) for native heavy-element embeddings."
    )
    return mapped, warn


def _deterministic_token_states(question: str, dim: int) -> tuple[list[str], torch.Tensor]:
    """Offline O(3)-invariant deterministic token embedding fallback when HuggingFace hub is unreachable."""
    words = re.findall(r"<vec>|[A-Za-z0-9_+.-]+|[^\sA-Za-z0-9_+.-]", question) or ["?"]
    tokens = ["[CLS]"] + words + ["[SEP]"]
    rows = []
    for pos_idx, tok in enumerate(tokens):
        digest = hashlib.sha256(f"ylm-v0-token|{tok.lower()}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big") % (2 ** 31 - 1)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.float32)
        # Add smooth positional phase so repeated words remain distinguishable by position.
        freqs = np.arange(dim, dtype=np.float32) / max(dim, 1)
        vec = vec + 0.15 * np.sin((pos_idx + 1) * (freqs * 6.2831853))
        vec /= max(float(np.linalg.norm(vec)), 1e-6)
        rows.append(vec)
    return tokens, torch.from_numpy(np.stack(rows, axis=0))


class _TextStateEncoder:
    """Encode a natural-language question into (tokens, Tensor[T, dim]) using cached HF weights or offline fallback."""

    def __init__(self, name: str, dim: int, allow_download: bool = False):
        self.name = name
        self.dim = int(dim)
        self.hf_tok = None
        self.hf_model = None
        self.using_fallback = False
        if name in V0_MODELS:
            repo_id = V0_MODELS[name]
            try:
                from transformers import AutoModel, AutoTokenizer
                self.hf_tok = AutoTokenizer.from_pretrained(repo_id, local_files_only=not allow_download)
                self.hf_model = AutoModel.from_pretrained(repo_id, local_files_only=not allow_download).eval()
            except Exception:
                self.using_fallback = True
        else:
            self.using_fallback = (name != "taskid")

    @torch.no_grad()
    def encode(self, question: str) -> tuple[list[str], torch.Tensor]:
        if self.hf_tok is not None and self.hf_model is not None:
            enc = self.hf_tok(question, return_tensors="pt")
            states = self.hf_model(**enc).last_hidden_state[0].detach().cpu()
            toks = self.hf_tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())
            return toks, states
        return _deterministic_token_states(question, self.dim)


def _channel_breakdown_by_irrep(e_tokens: torch.Tensor, irreps: o3.Irreps) -> dict[str, list[float]]:
    """Aggregate per-token L2 norms grouped by irrep label ('0e', '0o', '1o', '1e', '2e', ...)."""
    # e_tokens: (T, D)
    totals: dict[str, torch.Tensor] = {}
    cursor = 0
    for mul, ir in irreps:
        block_dim = mul * ir.dim
        seg = e_tokens[:, cursor:cursor + block_dim].reshape(e_tokens.shape[0], mul, ir.dim)
        norm_sq = seg.square().sum(dim=(-1, -2))
        key = str(ir)
        totals[key] = norm_sq if key not in totals else (totals[key] + norm_sq)
        cursor += block_dim
    return {k: v.clamp(min=0.0).sqrt().detach().cpu().tolist() for k, v in totals.items()}


def infer_v0_task(question: str) -> int:
    q = question.lower()
    dipole_kw = ("dipole", "polarity", "moment")
    force_kw = ("force", "gradient", "pull", "push", "escape", "transfer", "exit", "growth", "direction", "atom")
    if any(k in q for k in dipole_kw) and not any(k in q for k in force_kw):
        return 0
    return 1


@dataclass
class VectorPrediction:
    name: str
    vector_au: list[float]
    vector_physical: list[float]
    unit: str
    magnitude: float
    anchor_xyz_angstrom: list[float]
    anchor_atom: Optional[int] = None


@dataclass
class O3Verification:
    passed: bool
    reflection_tested: bool
    vector_equivariance_error: float
    invariance_error: float


@dataclass
class ProbeResult:
    engine: str
    question: str
    answer: str
    vectors: list[VectorPrediction]
    tokens: list[str]
    atom_attention: list[list[float]]
    pooled_atom_attention: list[float]
    channel_norms: dict[str, list[float]]
    domain: str = "general"
    domain_summary: dict = field(default_factory=dict)
    o3_check: Optional[O3Verification] = None
    warnings: list[str] = field(default_factory=list)
    frame: Optional[StructureFrame] = field(default=None, repr=False)

    @property
    def vectors_angstrom(self) -> list[list[float]]:
        return [v.vector_physical for v in self.vectors]

    @property
    def top_attended_atoms(self) -> list[dict]:
        if self.frame is None or not self.pooled_atom_attention:
            return []
        weights = np.asarray(self.pooled_atom_attention, dtype=np.float64)
        order = np.argsort(-weights)[: min(5, len(weights))]
        return [
            {
                "atom_index": int(i),
                "symbol": symbol_for_z(int(self.frame.Z[i])),
                "weight": float(weights[i]),
                "xyz_angstrom": [float(c) for c in self.frame.pos_angstrom[i]],
            }
            for i in order
        ]

    @property
    def l_channel_weights(self) -> dict[str, float]:
        totals = {k: float(np.mean(v)) for k, v in self.channel_norms.items() if len(v)}
        s = sum(totals.values())
        return {k: (val / s if s > 1e-12 else 0.0) for k, val in totals.items()}

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("frame", None)
        d["top_attended_atoms"] = self.top_attended_atoms
        d["l_channel_weights"] = self.l_channel_weights
        if self.frame is not None:
            d["structure"] = {
                "label": self.frame.label,
                "n_atoms": self.frame.n_atoms,
                "symbols": self.frame.symbols,
                "pos_angstrom": self.frame.pos_angstrom.tolist(),
                "cell_angstrom": None if self.frame.cell_angstrom is None else self.frame.cell_angstrom.tolist(),
            }
        return d

    def to_html(self, path: Optional[Union[str, Path]] = None, title: Optional[str] = None) -> str:
        from src.toolkit.viz import render_html_report
        html = render_html_report(self, title=title)
        if path is not None:
            out_p = Path(path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(html, encoding="utf-8")
        return html


class YLMProbe:
    """Unified directional probe wrapping either packaged v0 YLM checkpoints or YLMCausalLM runs."""

    def __init__(
        self,
        engine: str,
        model: torch.nn.Module,
        device: str = "cpu",
        *,
        v0_scale: Optional[torch.Tensor] = None,
        text_encoder: Optional[_TextStateEncoder] = None,
        causal_tokenizer: Optional[CharTokenizer] = None,
        source_path: str = "in-memory",
    ):
        self.engine = engine
        self.model = model.to(device).eval()
        self.device = device
        self.v0_scale = v0_scale.to(device) if v0_scale is not None else torch.ones(2, device=device)
        self.text_encoder = text_encoder
        self.causal_tokenizer = causal_tokenizer
        self.source_path = source_path

    @classmethod
    def default(cls, device: str = "cpu", allow_download: bool = False) -> "YLMProbe":
        for name in ("ylm_matscibert_s0.pt", "ylm_minilm_s0.pt", "baseline_taskid_s0.pt"):
            candidate = REPO_ROOT / "checkpoints" / "v0" / name
            if candidate.exists():
                return cls.from_checkpoint(candidate, device=device, allow_download=allow_download)
        raise FileNotFoundError("No default checkpoint found under checkpoints/v0/")

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        device: str = "cpu",
        allow_download: bool = False,
    ) -> "YLMProbe":
        p = Path(path)
        if p.is_dir() and (p / "config.json").exists():
            return cls.from_causal_lm(p, device=device)
        ck = torch.load(p, map_location="cpu")
        if isinstance(ck, dict) and "state_dict" in ck and "scale" in ck:
            text_name = ck.get("text", "matscibert")
            sd = ck["state_dict"]
            text_dim = int(sd["text_proj.0.weight"].shape[0])
            cfg = ck.get("args", {})
            tiny = (
                dict(hidden_irreps="8x0e + 2x0o + 4x1o + 2x1e + 2x2e", cross_irreps="4x0e + 4x1o + 2x1e + 2x2e")
                if cfg.get("tiny") else {}
            )
            model = YLM(
                text_dim=text_dim,
                text_mode="taskid" if text_name == "taskid" else "text",
                option=cfg.get("option", "A"),
                b_score_mode=cfg.get("b_score_mode", "legacy"),
                c_tp_mode=cfg.get("c_tp_mode", "fully_connected"),
                legacy_odd_scalars=cfg.get("legacy_odd_scalars", False),
                **tiny,
            )
            model.load_state_dict(sd, strict=True)
            scale = torch.as_tensor(ck["scale"], dtype=torch.float32)
            enc = _TextStateEncoder(text_name, text_dim, allow_download=allow_download)
            return cls("v0", model, device=device, v0_scale=scale, text_encoder=enc, source_path=str(p))
        raise ValueError(f"Unrecognized checkpoint format at {p}")

    @classmethod
    def from_causal_lm(
        cls,
        run_or_model: Union[str, Path, torch.nn.Module],
        tokenizer: Optional[CharTokenizer] = None,
        device: str = "cpu",
    ) -> "YLMProbe":
        if isinstance(run_or_model, (str, Path)):
            model, tok, _ = load_run(str(run_or_model), device=device)
            return cls("causal_lm", model, device=device, causal_tokenizer=tok, source_path=str(run_or_model))
        tok = tokenizer or CharTokenizer()
        return cls("causal_lm", run_or_model, device=device, causal_tokenizer=tok, source_path="in-memory")

    @torch.no_grad()
    def _ask_v0(self, frame: StructureFrame, question: str, task: Optional[int] = None) -> ProbeResult:
        warnings: list[str] = []
        chosen_task = infer_v0_task(question) if task is None else int(task)
        n_elem = int(self.model.encoder.embed.num_embeddings)
        mapped_z, warn_z = _map_elements_for_limit(frame.Z, n_elem)
        if warn_z:
            warnings.append(warn_z)
        if self.text_encoder is not None and self.text_encoder.using_fallback and self.model.text_mode != "taskid":
            warnings.append(
                f"Frozen HF encoder '{self.text_encoder.name}' not cached locally; using deterministic O(3)-invariant token projection."
            )

        param = next(self.model.parameters())
        dtype = param.dtype
        Z_t = torch.as_tensor(mapped_z[None, :], dtype=torch.long, device=self.device)
        pos_t = torch.as_tensor(frame.pos_bohr[None, :, :], dtype=dtype, device=self.device)
        atom_mask = torch.ones((1, frame.n_atoms), dtype=torch.bool, device=self.device)
        cell_t = (torch.as_tensor(frame.cell_bohr[None, :, :], dtype=dtype, device=self.device)
                  if frame.cell_bohr is not None else None)
        pbc_t = (torch.as_tensor(frame.pbc[None, :], dtype=torch.bool, device=self.device)
                 if frame.pbc is not None else None)

        if self.model.text_mode == "taskid":
            tokens = [f"[TASK={chosen_task}]"]
            text_t = self.model.task_embed(torch.tensor([chosen_task], device=self.device))[:, None, :]
        else:
            tokens, states = self.text_encoder.encode(question)
            text_t = states.to(device=self.device, dtype=dtype).unsqueeze(0)
        text_mask = torch.ones(text_t.shape[:2], dtype=torch.bool, device=self.device)

        h = self.model.text_proj(text_t)
        x = self.model.encoder(Z_t, pos_t, atom_mask, cell=cell_t, pbc=pbc_t)
        e, attn = self.model.cross(h, x, atom_mask)
        w = self.model.token_weight(h).squeeze(-1).masked_fill(~text_mask, float("-inf")).softmax(-1)
        e_pool = torch.einsum("bt,btd->bd", w, e)
        attn_pool = torch.einsum("bt,btn->bn", w, attn)[0]
        raw_vec = self.model.readout(e_pool)[0] * self.v0_scale[chosen_task].to(dtype)

        vec_au = raw_vec.detach().cpu().numpy().astype(np.float64)
        pooled_np = attn_pool.detach().cpu().numpy().astype(np.float64)
        top_atom = int(np.argmax(pooled_np)) if frame.n_atoms else None

        if chosen_task == 0:
            vec_phys = vec_au * AU_DIPOLE_TO_DEBYE
            unit = "Debye"
            name = "dipole"
            anchor = frame.pos_angstrom.mean(axis=0).tolist() if frame.n_atoms else [0.0, 0.0, 0.0]
            anchor_atom = None
            mag = float(np.linalg.norm(vec_phys))
            answer = (
                f"Predicted molecular dipole vector: ({vec_phys[0]:+.4f}, {vec_phys[1]:+.4f}, {vec_phys[2]:+.4f}) "
                f"{unit} (|mu| = {mag:.4f} {unit})."
            )
        else:
            vec_phys = vec_au * AU_FORCE_TO_EV_ANG
            unit = "eV/A"
            name = "force"
            anchor_atom = top_atom
            anchor = (frame.pos_angstrom[top_atom].tolist()
                      if top_atom is not None else [0.0, 0.0, 0.0])
            mag = float(np.linalg.norm(vec_phys))
            sym = symbol_for_z(int(frame.Z[top_atom])) if top_atom is not None else "?"
            answer = (
                f"Routed to {sym}#{top_atom} (attention={pooled_np[top_atom]:.3f}): predicted 3D vector "
                f"({vec_phys[0]:+.4f}, {vec_phys[1]:+.4f}, {vec_phys[2]:+.4f}) {unit} (|F| = {mag:.4f} {unit})."
            )

        vp = VectorPrediction(
            name=name,
            vector_au=vec_au.tolist(),
            vector_physical=vec_phys.tolist(),
            unit=unit,
            magnitude=mag,
            anchor_xyz_angstrom=anchor,
            anchor_atom=anchor_atom,
        )
        return ProbeResult(
            engine="v0",
            question=question,
            answer=answer,
            vectors=[vp],
            tokens=tokens,
            atom_attention=attn[0].detach().cpu().tolist(),
            pooled_atom_attention=pooled_np.tolist(),
            channel_norms=_channel_breakdown_by_irrep(e[0], self.model.cross.out_irreps),
            warnings=warnings,
            frame=frame,
        )

    @torch.no_grad()
    def _ask_causal_lm(self, frame: StructureFrame, question: str, max_new: int = 32) -> ProbeResult:
        tok = self.causal_tokenizer
        geom = frame.to_geom_dict()
        warnings: list[str] = []
        param = next(self.model.parameters())
        dtype = param.dtype

        prompt_ex = Example([Seg(question)], geom)
        if self.model.vec_head is not None:
            decoded = decode_with_vectors(self.model, tok, [prompt_ex], max_new=max_new, device=self.device)[0]
            gen_text = decoded["text"]
            raw_vecs = [np.asarray(v, dtype=np.float64) for v in decoded["vectors"]]
        else:
            from src.tasks.common import decode_greedy
            gen_text = decode_greedy(self.model, tok, [prompt_ex], max_new=max_new, device=self.device)[0]
            raw_vecs = []

        # Ensure an explicit <vec> probe readout is always available when vec_head is enabled.
        full_text = question + gen_text
        if self.model.vec_head is not None and not raw_vecs:
            probe_text = full_text if VEC_TOKEN in full_text else (question.rstrip() + " " + VEC_TOKEN)
            batch = collate([Example([Seg(probe_text)], geom)], tok, max_len=min(256, self.model.cfg.n_positions),
                            device=self.device, dtype=dtype)
            out = self.model(**model_inputs(batch), output_routing=True)
            if "vec" in out and out["vec"].numel() >= 3:
                raw_vecs = [out["vec"][-1].detach().cpu().numpy().astype(np.float64)]
                if VEC_TOKEN not in gen_text:
                    gen_text = (gen_text + " " + VEC_TOKEN).strip()
        else:
            probe_text = full_text if full_text else question
            batch = collate([Example([Seg(probe_text)], geom)], tok, max_len=min(256, self.model.cfg.n_positions),
                            device=self.device, dtype=dtype)
            out = self.model(**model_inputs(batch), output_routing=True)

        ids = batch["input_ids"][0].detach().cpu().tolist()
        token_labels = [tok.itos[i] if hasattr(tok, "itos") else tok.decode([i]) for i in ids]
        n_atoms = frame.n_atoms
        if "atom_attention" in out:
            attn_mat = out["atom_attention"][0, :, :n_atoms].detach().cpu().numpy().astype(np.float64)
            pooled = attn_mat.mean(axis=0)
        else:
            attn_mat = np.full((len(token_labels), max(n_atoms, 1)), 1.0 / max(n_atoms, 1), dtype=np.float64)
            pooled = attn_mat.mean(axis=0)

        if "side_irreps" in out and len(self.model.adapters) > 0:
            ch_norms = _channel_breakdown_by_irrep(out["side_irreps"][0], self.model.adapters[0].cross.out_irreps)
        else:
            ch_norms = {"0e": [1.0] * len(token_labels)}

        top_atom = int(np.argmax(pooled)) if n_atoms else None
        anchor = frame.pos_angstrom[top_atom].tolist() if top_atom is not None else [0.0, 0.0, 0.0]
        vpreds = []
        for idx, rv in enumerate(raw_vecs):
            mag = float(np.linalg.norm(rv))
            vpreds.append(VectorPrediction(
                name=f"vec_{idx}",
                vector_au=rv.tolist(),
                vector_physical=rv.tolist(),
                unit="eV/A",
                magnitude=mag,
                anchor_xyz_angstrom=anchor,
                anchor_atom=top_atom,
            ))

        return ProbeResult(
            engine="causal_lm",
            question=question,
            answer=gen_text,
            vectors=vpreds,
            tokens=token_labels,
            atom_attention=attn_mat.tolist(),
            pooled_atom_attention=pooled.tolist(),
            channel_norms=ch_norms,
            warnings=warnings,
            frame=frame,
        )

    def ask(
        self,
        structure: Union[str, Path, dict, StructureFrame],
        question: str,
        *,
        domain: str = "general",
        task: Optional[int] = None,
        verify_o3: bool = False,
        max_new: int = 32,
    ) -> ProbeResult:
        frame = load_structure(structure)
        res = (self._ask_v0(frame, question, task=task)
               if self.engine == "v0"
               else self._ask_causal_lm(frame, question, max_new=max_new))
        res.domain = domain
        if domain != "general":
            from src.toolkit.domains import enrich_domain_result
            enrich_domain_result(self, frame, res, domain=domain)
        if verify_o3:
            res.o3_check = self.verify_equivariance(frame, question, task=task)
        return res

    @torch.no_grad()
    def verify_equivariance(
        self,
        structure: Union[str, Path, dict, StructureFrame],
        question: str = "What is the force vector on the first atom?",
        *,
        reflection: bool = True,
        task: Optional[int] = None,
        seed: int = 17,
        tol: float = 2e-4,
    ) -> O3Verification:
        """Apply a random improper/proper O(3) transformation + translation and verify exact covariance."""
        frame = load_structure(structure)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            R_t = o3.rand_matrix(dtype=torch.float64)
        if reflection:
            R_t = -R_t
        R = R_t.numpy()
        shift = np.array([3.25, -1.75, 2.10], dtype=np.float64)
        moved = frame.transformed(R, shift)

        r_orig = self._ask_v0(frame, question, task=task) if self.engine == "v0" else self._ask_causal_lm(frame, question, max_new=8)
        r_rot = self._ask_v0(moved, question, task=task) if self.engine == "v0" else self._ask_causal_lm(moved, question, max_new=8)

        vec_err = 0.0
        for v0, v1 in zip(r_orig.vectors, r_rot.vectors):
            expected = np.asarray(v0.vector_physical, dtype=np.float64) @ R.T
            actual = np.asarray(v1.vector_physical, dtype=np.float64)
            denom = max(float(np.linalg.norm(expected)), 1.0)
            vec_err = max(vec_err, float(np.linalg.norm(actual - expected)) / denom)

        a0 = np.asarray(r_orig.pooled_atom_attention, dtype=np.float64)
        a1 = np.asarray(r_rot.pooled_atom_attention, dtype=np.float64)
        inv_err = float(np.max(np.abs(a0 - a1))) if a0.size and a0.shape == a1.shape else 0.0

        return O3Verification(
            passed=(vec_err <= tol and inv_err <= tol),
            reflection_tested=reflection,
            vector_equivariance_error=vec_err,
            invariance_error=inv_err,
        )


def verify_o3_equivariance(
    structure: Union[str, Path, dict, StructureFrame],
    question: str = "What is the force vector on the oxygen atom?",
    checkpoint: Optional[Union[str, Path]] = None,
) -> O3Verification:
    probe = YLMProbe.from_checkpoint(checkpoint) if checkpoint else YLMProbe.default()
    return probe.verify_equivariance(structure, question)

"""CPU inference API for the packaged v0 checkpoints: (molecule, question) -> 3D vector.

    from src.api import YLMPredictor
    p = YLMPredictor()                         # ylm_matscibert_s0 + frozen MatSciBERT
    p.predict("What is the force on the oxygen atom?", smiles="CCO")

CLI: python -m src.api --question "Give the molecular dipole vector." --smiles CCO
HTTP (optional, needs fastapi + uvicorn): uvicorn src.api:app --port 8000  ->  POST /predict

Scope: TensorQA-v0 only (QM9-size H/C/N/O/F molecules, GFN2-xTB labels, dipole or per-atom
force). This is not a protocol generator and not a general force field.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src.model import YLM  # import before e3nn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "checkpoints" / "v0" / "ylm_matscibert_s0.pt"
TEXT_MODELS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "matscibert": "m3rg-iitd/matscibert"}
ELEMENTS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
TASKS = {"dipole": 0, "force": 1}
UNITS = {"dipole": ("e*bohr", "Debye", 2.541746), "force": ("hartree/bohr", "eV/A", 51.42208619)}
BOHR_PER_ANGSTROM = 1.8897261246
MAX_ATOMS = 64
MAX_QUESTION_CHARS = 300


class InputError(ValueError):
    """Bad request: the caller should fix the input."""


FORCE_WORDS = ("force", "pull", "push", "atom")


def infer_task(question: str) -> str:
    """'dipole' or 'force' from the wording; anything else (e.g. a protocol question) is rejected."""
    q = question.lower()
    if "dipole" in q:
        return "dipole"
    if any(w in q for w in FORCE_WORDS):
        return "force"
    raise InputError("v0 answers dipole or per-atom force questions only; pass task= to override")


@lru_cache(maxsize=256)
def smiles_to_mol(smiles: str, seed: int = 0):
    """RDKit molecule with hydrogens and one ETKDG + MMFF conformer (angstrom). Cached."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InputError(f"invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    if mol.GetNumAtoms() > MAX_ATOMS:
        raise InputError(f"{mol.GetNumAtoms()} atoms with hydrogens; the limit is {MAX_ATOMS}")
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise InputError(f"could not embed a 3D conformer for {smiles!r}")
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


class HFTextEncoder:
    """Frozen Hugging Face encoder -> token states (T, d) including [CLS]/[SEP], as in training."""

    def __init__(self, name: str, int8: bool = False):
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging

        logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval()
        if int8:
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)

    @torch.inference_mode()
    def __call__(self, question: str) -> torch.Tensor:
        enc = self.tokenizer(question, return_tensors="pt", truncation=True, max_length=128)
        return self.model(**enc).last_hidden_state[0].float()


class YLMPredictor:
    """Load once, then call predict() per request. Question embeddings are cached (LRU)."""

    def __init__(self, ckpt=DEFAULT_CKPT, *, threads: int = 1, int8_text: bool = False,
                 text_encoder=None, cache_size: int = 1024):
        if threads:
            torch.set_num_threads(threads)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)  # trusted repo checkpoint
        self.text_name = ck.get("text", "minilm")
        if self.text_name not in TEXT_MODELS:
            raise ValueError(f"unsupported checkpoint text encoder {self.text_name!r}")
        cfg = ck.get("args", {})
        tiny = (dict(hidden_irreps="8x0e + 2x0o + 4x1o + 2x1e + 2x2e", cross_irreps="4x0e + 4x1o + 2x1e + 2x2e")
                if cfg.get("tiny") else {})
        self.model = YLM(text_dim=ck["state_dict"]["text_proj.0.weight"].shape[0], text_mode="text",
                         option=cfg.get("option", "A"), b_score_mode=cfg.get("b_score_mode", "legacy"),
                         c_tp_mode=cfg.get("c_tp_mode", "fully_connected"),
                         legacy_odd_scalars=cfg.get("legacy_odd_scalars", False),
                         n_cross_layers=cfg.get("n_cross_layers", 1), **tiny)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.scale = torch.as_tensor(ck["scale"], dtype=torch.float32)
        self.name = Path(ckpt).stem + ("+int8text" if int8_text else "")
        self.encode = text_encoder or HFTextEncoder(TEXT_MODELS[self.text_name], int8=int8_text)
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.cache_size = cache_size
        self.predict("Give the molecular dipole vector.", atomic_numbers=[8, 1, 1],  # warm up
                     positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

    def text_states(self, question: str) -> torch.Tensor:
        if question in self._cache:
            self._cache.move_to_end(question)
            return self._cache[question]
        states = self.encode(question)
        self._cache[question] = states
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return states

    def _check(self, question, atomic_numbers, positions, smiles, unit, task):
        if not isinstance(question, str) or not question.strip():
            raise InputError("question must be a nonempty string")
        if len(question) > MAX_QUESTION_CHARS:
            raise InputError(f"question longer than {MAX_QUESTION_CHARS} characters")
        if task is not None and task not in TASKS:
            raise InputError(f"task must be one of {sorted(TASKS)} or omitted")
        if (smiles is None) == (atomic_numbers is None):
            raise InputError("give exactly one of smiles or atomic_numbers + positions")
        mol = None
        if smiles is not None:
            mol = smiles_to_mol(smiles)
            z = [a.GetAtomicNum() for a in mol.GetAtoms()]
            pos, unit = mol.GetConformer().GetPositions(), "angstrom"
        else:
            try:
                z, pos = [int(a) for a in atomic_numbers], np.asarray(positions, dtype=np.float64)
            except (TypeError, ValueError) as err:
                raise InputError("atomic_numbers must be integers and positions numbers") from err
            if pos.shape != (len(z), 3):
                raise InputError("positions must have shape (n_atoms, 3)")
        if unit not in ("angstrom", "bohr"):
            raise InputError("unit must be 'angstrom' or 'bohr'")
        if not 1 <= len(z) <= MAX_ATOMS:
            raise InputError(f"need 1..{MAX_ATOMS} atoms, got {len(z)}")
        bad = sorted({int(a) for a in z} - set(ELEMENTS))
        if bad:
            raise InputError(f"unsupported elements {bad}; v0 covers H, C, N, O, F")
        if not np.isfinite(pos).all():
            raise InputError("positions must be finite")
        pos = pos * (BOHR_PER_ANGSTROM if unit == "angstrom" else 1.0)
        return [int(a) for a in z], pos, task or infer_task(question), mol

    @torch.inference_mode()
    def predict(self, question: str, *, atomic_numbers=None, positions=None, smiles=None,
                unit: str = "angstrom", task: str | None = None) -> dict:
        t0 = time.perf_counter()
        z, pos_bohr, task, mol = self._check(question, atomic_numbers, positions, smiles, unit, task)
        text = self.text_states(question)[None]
        out, attn = self.model(Z=torch.tensor([z]), pos=torch.tensor(pos_bohr, dtype=torch.float32)[None],
                               atom_mask=torch.ones(1, len(z), dtype=torch.bool), text=text,
                               text_mask=torch.ones(text.shape[:2], dtype=torch.bool),
                               task=torch.tensor([TASKS[task]]))
        vec = (out[0] * self.scale[TASKS[task]]).tolist()
        unit_au, unit_phys, factor = UNITS[task]
        weights = attn[0].tolist()
        top = sorted(range(len(z)), key=lambda i: -weights[i])[:3]
        result = {
            "task": task,
            "vector": vec,
            "unit": unit_au,
            "vector_physical": [v * factor for v in vec],
            "unit_physical": unit_phys,
            "routed_atom": top[0] if task == "force" else None,
            "top_attention": [{"atom": i, "element": ELEMENTS[z[i]], "weight": weights[i]} for i in top],
            "atomic_numbers": z,
            "model": self.name,
        }
        if mol is not None:
            from src.parse_question import resolve

            parsed = resolve(question, mol) if task == "force" else -1
            result["parsed_atom"] = parsed if parsed >= 0 else None
            result["positions_angstrom"] = (pos_bohr / BOHR_PER_ANGSTROM).tolist()
        result["latency_ms"] = (time.perf_counter() - t0) * 1e3
        return result


def create_app(predictor: YLMPredictor | None = None):
    """Small FastAPI app: POST /predict with the predict() keyword arguments as JSON."""
    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(title="YLM v0 CPU API")
    state = {"predictor": predictor}
    lock = threading.Lock()  # one request at a time: parallel torch calls would fight over cores

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/predict")
    def predict(req: dict = Body(...)):
        if state["predictor"] is None:
            state["predictor"] = YLMPredictor(os.environ.get("YLM_CKPT", DEFAULT_CKPT))
        allowed = {"question", "smiles", "atomic_numbers", "positions", "unit", "task"}
        try:
            if not isinstance(req, dict) or set(req) - allowed:
                raise InputError(f"request keys must be a subset of {sorted(allowed)}")
            req = dict(req)
            with lock:
                return state["predictor"].predict(req.pop("question", ""), **req)
        except InputError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err

    return app


def __getattr__(name):  # `uvicorn src.api:app` builds the app lazily
    if name == "app":
        return create_app()
    raise AttributeError(name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--question", required=True)
    ap.add_argument("--smiles")
    ap.add_argument("--xyz", help="XYZ file (angstrom)")
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--int8-text", action="store_true", help="int8 dynamic quantization of the text encoder")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)
    kwargs = {"smiles": args.smiles, "task": args.task}
    if args.xyz:
        from rdkit.Chem import GetPeriodicTable

        lines = Path(args.xyz).read_text().split("\n")
        rows = [ln.split() for ln in lines[2:2 + int(lines[0])]]
        table = GetPeriodicTable()
        kwargs.update(smiles=None, atomic_numbers=[table.GetAtomicNumber(r[0]) for r in rows],
                      positions=[[float(v) for v in r[1:4]] for r in rows])
    predictor = YLMPredictor(args.ckpt, threads=args.threads, int8_text=args.int8_text)
    try:
        print(json.dumps(predictor.predict(args.question, **kwargs), indent=2))
    except InputError as err:
        raise SystemExit(f"error: {err}") from err


if __name__ == "__main__":
    main()

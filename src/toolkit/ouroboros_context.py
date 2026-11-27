"""Strict reader for ouroboros-ocsr `ouroboros.molecule_context/v1` JSONL (ouroboros-ocsr issue #77).

Each row becomes one `ContextDecision`:
    status=ok        -> use the SMILES
    status=review    -> "ask a human"; no SMILES is passed on
    rejected/invalid -> dropped
A `gold` field or any key outside the frozen v1 fields raises ValueError, so answer
labels can never leak into the prompt. Schema check, SMILES and flags for `ok` rows come
from `lab.molecule_context.read_context_jsonl`; 3D frames stay in `ouroboros_bridge.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional, Union

from lab.molecule_context import MoleculeRef, read_context_jsonl

ASK_HUMAN = "ask a human"
USE_SMILES = "use smiles"
FROZEN_KEYS = frozenset({
    "schema", "id", "status", "usable", "smiles", "smiles_no_stereo", "pred_raw", "formula",
    "inchikey", "molfile", "heavy_atoms", "pseudo_atoms", "reasons", "stereo_specified",
    "stereo_unassigned", "subset", "gate",
})
FROZEN_GATE_KEYS = frozenset({"accepted", "disagreement", "margin", "rejection_reason"})


@dataclass
class ContextDecision:
    id: str
    status: str
    action: str                          # USE_SMILES or ASK_HUMAN
    smiles: Optional[str] = None         # set only for status=ok
    reasons: list[str] = field(default_factory=list)
    ref: Optional[MoleculeRef] = None    # prompt-ready ref for status=ok


def check_row(rec: dict, where: str = "row") -> None:
    """Raise ValueError on a `gold` field or any key that is not in the frozen v1 schema."""
    gate = rec.get("gate")
    if "gold" in rec or (isinstance(gate, dict) and "gold" in gate):
        raise ValueError(f"{where}: 'gold' field is not allowed in molecule context")
    unknown = set(rec) - FROZEN_KEYS
    if isinstance(gate, dict):
        unknown |= {f"gate.{k}" for k in set(gate) - FROZEN_GATE_KEYS}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")


def read_molecule_context(path: Union[str, Path]) -> list[ContextDecision]:
    """Read and check a context JSONL file; returns decisions for ok and review rows, in file order."""
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            rec = json.loads(line)
            check_row(rec, f"{path}:{i}")
            rows.append(rec)
    ok_refs = {r.source_id: r for r in read_context_jsonl(path, only_usable=True)}
    out = []
    for rec in rows:
        rid, status = str(rec.get("id", "")), rec.get("status")
        if status == "ok" and rid in ok_refs:
            ref = ok_refs[rid]
            out.append(ContextDecision(rid, status, USE_SMILES, ref.smiles, [], ref))
        elif status == "review":
            out.append(ContextDecision(rid, status, ASK_HUMAN, None, list(rec.get("reasons") or [])))
    return out

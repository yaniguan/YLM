"""Map OCSR output ("eyes") to the reasoner's molecule context ("brain").

Input (preferred): the context JSONL from ouroboros-ocsr `lab/molecule_context.py`
(schema `ouroboros.molecule_context/v1`); see `read_context_jsonl`.
Input (raw): rows from ouroboros-ocsr `predictions.csv` (same columns that
`src/toolkit/ouroboros_bridge.load_ouroboros_predictions_csv` reads):
    id | image_id | source_id, smiles | pred | prediction, accepted, margin, source_image

Output: `MoleculeRef` objects and a short text block that goes in front of a
protocol question (see `lab.protocol_qa.build_prompt`).

Stdlib only. For 3D frames (chirality, forces) use the existing bridge in
`src/toolkit/ouroboros_bridge.py`; this module only builds text context.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

_TRUE = {"1", "true", "yes", "t", ""}


@dataclass
class MoleculeRef:
    ref: str                       # short tag used in the prompt, e.g. "M1"
    smiles: str
    source_id: str
    accepted: bool = True
    margin: Optional[float] = None
    source_image: str = ""
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def has_stereo(smiles: str) -> bool:
    """True if the SMILES carries tetrahedral (@) or double-bond (/ \\) stereo marks."""
    return any(ch in smiles for ch in "@/\\")


def rows_to_refs(
    rows: Iterable[dict],
    *,
    min_margin: Optional[float] = None,
    drop_rejected: bool = False,
) -> list[MoleculeRef]:
    """Convert OCSR rows to MoleculeRefs. Low-confidence rows are kept but flagged."""
    refs: list[MoleculeRef] = []
    for idx, row in enumerate(rows):
        smi = (row.get("smiles") or row.get("pred") or row.get("prediction") or "").strip()
        if not smi:
            continue
        accepted = str(row.get("accepted", "true")).strip().lower() in _TRUE
        if drop_rejected and not accepted:
            continue
        m_raw = row.get("margin")
        margin = float(m_raw) if m_raw not in (None, "") else None
        flags = []
        if not accepted:
            flags.append("ocsr_rejected")
        if min_margin is not None and margin is not None and margin < min_margin:
            flags.append("low_margin")
        if has_stereo(smi):
            flags.append("stereo")
        src = str(row.get("id") or row.get("image_id") or row.get("source_id") or f"row_{idx}")
        refs.append(
            MoleculeRef(
                ref=f"M{len(refs) + 1}",
                smiles=smi,
                source_id=src,
                accepted=accepted,
                margin=margin,
                source_image=str(row.get("source_image") or row.get("image") or ""),
                flags=flags,
            )
        )
    return refs


def read_predictions_csv(source: Union[str, Path, io.TextIOBase], **kw) -> list[MoleculeRef]:
    """Read an OCSR predictions.csv path or open text stream."""
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8", newline="") as fh:
            return rows_to_refs(csv.DictReader(fh), **kw)
    return rows_to_refs(csv.DictReader(source), **kw)


OCSR_SCHEMA = "ouroboros.molecule_context/v1"


def read_context_jsonl(path: Union[str, Path], *, only_usable: bool = True) -> list[MoleculeRef]:
    """Read ouroboros-ocsr `lab/molecule_context.py` JSONL (schema v1).

    Only `status=ok` rows carry a trusted structure. With `only_usable=False`,
    `review` rows are kept and flagged; `rejected` / `invalid` rows are always dropped.
    """
    refs: list[MoleculeRef] = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("schema") != OCSR_SCHEMA:
            raise ValueError(f"{path}:{i}: schema must be {OCSR_SCHEMA}, got {rec.get('schema')!r}")
        status = rec.get("status")
        if status not in ("ok", "review") or (only_usable and status != "ok") or not rec.get("smiles"):
            continue
        flags = [] if status == "ok" else ["review"] + list(rec.get("reasons") or [])
        if has_stereo(rec["smiles"]):
            flags.append("stereo")
        gate = rec.get("gate") or {}
        refs.append(MoleculeRef(ref=f"M{len(refs) + 1}", smiles=rec["smiles"], source_id=str(rec.get("id", "")),
                                accepted=status == "ok", margin=gate.get("margin"), flags=flags))
    return refs


def context_block(refs: Iterable[MoleculeRef]) -> str:
    """Text block for the prompt. Flags tell the reasoner which structures to doubt."""
    refs = list(refs)
    if not refs:
        return ""
    lines = ["Molecules (from OCSR; check flagged ones):"]
    for r in refs:
        flag = f" [{', '.join(r.flags)}]" if r.flags else ""
        lines.append(f"- {r.ref} ({r.source_id}): {r.smiles}{flag}")
    return "\n".join(lines) + "\n"

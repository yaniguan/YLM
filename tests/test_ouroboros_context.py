"""ouroboros-ocsr molecule_context/v1 JSONL -> YLM prompt (ouroboros-ocsr issue #77)."""
import json
from pathlib import Path

import pytest

from lab import molecule_context as mc, protocol_qa as pqa
from src.toolkit.ouroboros_context import ASK_HUMAN, USE_SMILES, read_molecule_context

LAB_FIX = Path(__file__).resolve().parents[1] / "lab" / "fixtures"
CONTEXT = LAB_FIX / "ocsr_context_tiny.jsonl"   # real ouroboros-ocsr output (ok, review, rejected, invalid rows)


def test_molecule_context_jsonl_to_prompt(tmp_path):
    dec = read_molecule_context(CONTEXT)
    assert [(d.id, d.action) for d in dec] == [
        ("aspirin", USE_SMILES), ("acetate_salt", USE_SMILES), ("alanine_flat", ASK_HUMAN),
        ("l_alanine", USE_SMILES), ("r_group", ASK_HUMAN),
    ]
    by_id = {d.id: d for d in dec}
    assert by_id["l_alanine"].smiles == "C[C@H](N)C(=O)O"
    assert by_id["r_group"].smiles is None and by_id["r_group"].reasons == ["pseudo_atoms"]

    block = mc.context_block(d.ref for d in dec if d.action == USE_SMILES)
    prompt = pqa.build_prompt(pqa.load_jsonl(LAB_FIX / "protocol_qa_tiny.jsonl")[0], block)
    assert "C[C@H](N)C(=O)O" in prompt and "[R1]" not in prompt
    assert prompt.index("Molecules") < prompt.index("Question:")

    row = json.loads(CONTEXT.read_text().splitlines()[0])
    for extra, msg in [({"gold": "CCO"}, "gold"), ({"label": 1}, "unknown"),
                       ({"gate": dict(row["gate"], gold="CCO")}, "gold")]:
        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps(dict(row, **extra)) + "\n")
        with pytest.raises(ValueError, match=msg):
            read_molecule_context(bad)

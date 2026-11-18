"""Tests for `src/provenance.py` and the `provenance` block in `results.json`."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pickle
import sys

import e3nn
import pytest
import torch

from src.provenance import (
    OFFICIAL_V0_QUESTION_COUNT,
    OFFICIAL_V0_QUESTIONS_CANONICAL_SHA256,
    OFFICIAL_V0_SPLIT_COUNTS,
    OFFICIAL_V0_TARBALL_SHA256,
    OFFICIAL_V0_UNIQUE_STRINGS,
    OFFICIAL_V0_UNPACKED_JSON_SHA256,
    REPO_ROOT,
    build_provenance,
    file_sha256,
    verify_official_v0_questions,
)


def _load_script(rel_path: str, mod_name: str):
    path = Path(__file__).resolve().parents[1] / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_provenance_keys_and_hash_changes_on_question_edit(tmp_path: Path):
    q_file = tmp_path / "tensorqa_v0.json"
    t_file = tmp_path / "text_minilm.pt"
    q_file.write_text(json.dumps([{"mol": 0, "split": "train", "question": "What is the dipole?"}]), encoding="utf-8")
    torch.save({"dim": 8, "states": {"What is the dipole?": torch.ones(2, 8)}}, t_file)

    prov1 = build_provenance(
        questions_path=q_file,
        text_cache_path=t_file,
        split_counts={"train": 4, "val": 2, "test": 2, "test_heldout": 2},
        eval_rotations=4,
        eval_reflections=True,
        eval_seed=0,
    )
    expected_keys = {
        "questions_path",
        "questions_sha256",
        "text_cache_path",
        "text_cache_sha256",
        "split_counts",
        "eval_rotations",
        "eval_reflections",
        "eval_seed",
        "git_commit",
        "torch_version",
        "e3nn_version",
    }
    assert set(prov1.keys()) == expected_keys
    assert len(prov1["questions_sha256"]) == 64
    assert len(prov1["text_cache_sha256"]) == 64
    assert prov1["split_counts"] == {"train": 4, "val": 2, "test": 2, "test_heldout": 2}
    assert prov1["eval_rotations"] == 4
    assert prov1["eval_reflections"] is True
    assert prov1["eval_seed"] == 0
    assert prov1["torch_version"] == torch.__version__
    assert prov1["e3nn_version"] == e3nn.__version__

    # Changing the question file must change `questions_sha256`
    q_file.write_text(
        json.dumps([{"mol": 0, "split": "train", "question": "Give the dipole vector."}]),
        encoding="utf-8",
    )
    prov2 = build_provenance(
        questions_path=q_file,
        text_cache_path=t_file,
        split_counts={"train": 4, "val": 2, "test": 2, "test_heldout": 2},
        eval_rotations=4,
        eval_reflections=True,
    )
    assert prov2["questions_sha256"] != prov1["questions_sha256"]
    assert prov2["text_cache_sha256"] == prov1["text_cache_sha256"]
    assert file_sha256(tmp_path / "does_not_exist.json") is None


def test_04_train_and_16_train_quad_write_provenance_block(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    labels = [
        {
            "numbers": [8, 1, 1],
            "pos": [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.4, 1.7, 0.0]],
            "dipole": [0.1, 0.2, 0.0],
            "forces": [[0.01, -0.02, 0.0], [-0.005, 0.01, 0.0], [-0.005, 0.01, 0.0]],
        },
        {
            "numbers": [6, 8],
            "pos": [[0.0, 0.0, 0.0], [2.1, 0.0, 0.0]],
            "dipole": [-0.3, 0.0, 0.0],
            "forces": [[0.02, 0.0, 0.0], [-0.02, 0.0, 0.0]],
        },
    ]
    with open(data_dir / "qm9_xtb.pkl", "wb") as handle:
        pickle.dump(labels, handle)

    q1 = "What is the dipole?"
    q2 = "What is the force on atom 0?"
    questions = []
    for sp in ("train", "val", "test", "test_heldout"):
        questions.append({"mol": 0, "split": sp, "task": "dipole", "atom": -1, "question": q1})
        questions.append({"mol": 1, "split": sp, "task": "force", "atom": 0, "question": q2})
    q_path = data_dir / "tensorqa_v0.json"
    q_path.write_text(json.dumps(questions), encoding="utf-8")

    text_cache = {
        "dim": 8,
        "states": {
            q1: torch.randn(2, 8),
            q2: torch.randn(3, 8),
        },
    }
    t_path = data_dir / "text_minilm.pt"
    torch.save(text_cache, t_path)

    train04 = _load_script("scripts/04_train.py", "train04_prov")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "04_train.py",
            "--text",
            "minilm",
            "--name",
            "tiny_prov",
            "--tiny",
            "--epochs",
            "1",
            "--max_steps",
            "1",
            "--batch_size",
            "2",
            "--eval_rotations",
            "2",
            "--eval_reflections",
            "--device",
            "cpu",
        ],
    )
    train04.main()

    saved04 = json.loads((tmp_path / "runs" / "tiny_prov" / "results.json").read_text(encoding="utf-8"))
    prov04 = saved04["provenance"]
    assert prov04["questions_sha256"] == file_sha256(q_path)
    assert prov04["text_cache_sha256"] == file_sha256(t_path)
    assert prov04["split_counts"] == {"train": 2, "val": 2, "test": 2, "test_heldout": 2}
    assert prov04["eval_rotations"] == 2
    assert prov04["eval_reflections"] is True
    assert prov04["eval_seed"] == 0
    assert prov04["torch_version"] == torch.__version__
    assert prov04["e3nn_version"] == e3nn.__version__

    # Also check `16_train_quad.py` writes `provenance`
    geom = [{"numbers": [6, 1], "pos": [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]} for _ in range(10)]
    q_mat = [[1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, -0.5]]
    quad = {mol: q_mat for mol in range(10)}
    with open(data_dir / "quad_geom.pkl", "wb") as handle:
        pickle.dump(geom, handle)
    with open(data_dir / "qm9_quad.pkl", "wb") as handle:
        pickle.dump(quad, handle)

    train16 = _load_script("scripts/16_train_quad.py", "train16_prov")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "16_train_quad.py",
            "--name",
            "quad_prov",
            "--geom",
            "data/quad_geom.pkl",
            "--quad",
            "data/qm9_quad.pkl",
            "--questions",
            "data/missing_questions.json",
            "--tiny",
            "--epochs",
            "1",
            "--max_steps",
            "1",
            "--batch_size",
            "4",
            "--eval_rotations",
            "2",
            "--device",
            "cpu",
        ],
    )
    train16.main()
    saved16 = json.loads((tmp_path / "runs" / "quad_prov" / "results.json").read_text(encoding="utf-8"))
    prov16 = saved16["provenance"]
    assert prov16["questions_sha256"] is None
    assert prov16["split_counts"] == {"train": 8, "val": 1, "test": 1}
    assert prov16["eval_rotations"] == 2
    assert prov16["eval_reflections"] is False


def test_verify_official_v0_rejects_wrong_counts_or_hash(tmp_path: Path, monkeypatch):
    # Wrong total question count fails immediately
    bad_count = tmp_path / "wrong_count.json"
    bad_count.write_text(
        json.dumps([{"mol": 0, "split": "train", "task": "dipole", "atom": -1, "question": "q0"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Question count mismatch"):
        verify_official_v0_questions(bad_count)

    # Matching counts/unique strings with a modified question text fails the pinned hash check
    import src.provenance as prov_mod

    monkeypatch.setattr(prov_mod, "OFFICIAL_V0_QUESTION_COUNT", 4)
    monkeypatch.setattr(prov_mod, "OFFICIAL_V0_UNIQUE_STRINGS", 2)
    monkeypatch.setattr(prov_mod, "OFFICIAL_V0_SPLIT_COUNTS", {"train": 1, "val": 1, "test": 1, "test_heldout": 1})
    tampered = tmp_path / "tampered.json"
    tampered.write_text(
        json.dumps([
            {"mol": 0, "split": "train", "task": "dipole", "atom": -1, "question": "q_a"},
            {"mol": 0, "split": "val", "task": "dipole", "atom": -1, "question": "q_a"},
            {"mol": 0, "split": "test", "task": "force", "atom": 0, "question": "q_b"},
            {"mol": 0, "split": "test_heldout", "task": "force", "atom": 0, "question": "q_b"},
        ]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Official v0 question hash mismatch"):
        verify_official_v0_questions(tampered)

    # DATA_CARD.md must document the pinned hashes and counts
    card = (REPO_ROOT / "DATA_CARD.md").read_text(encoding="utf-8")
    assert OFFICIAL_V0_TARBALL_SHA256 in card
    assert OFFICIAL_V0_UNPACKED_JSON_SHA256 in card
    assert OFFICIAL_V0_QUESTIONS_CANONICAL_SHA256 in card
    assert "84,757" in card and "584" in card


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "tensorqa_v0.json").exists()
    and not (REPO_ROOT / "data" / "release" / "tensorqa_v0.tar.gz").exists(),
    reason="Official TensorQA-v0 data file not present in this checkout.",
)
def test_official_v0_file_matches_pinned_hash_when_present():
    q_json = REPO_ROOT / "data" / "tensorqa_v0.json"
    tar_gz = REPO_ROOT / "data" / "release" / "tensorqa_v0.tar.gz"
    target = q_json if q_json.exists() else tar_gz
    info = verify_official_v0_questions(target)
    assert info["n_questions"] == OFFICIAL_V0_QUESTION_COUNT
    assert info["n_unique_strings"] == OFFICIAL_V0_UNIQUE_STRINGS
    assert info["split_counts"] == OFFICIAL_V0_SPLIT_COUNTS
    assert info["canonical_questions_sha256"] == OFFICIAL_V0_QUESTIONS_CANONICAL_SHA256


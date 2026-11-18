"""Record data hashes, split counts, eval settings, and official TensorQA-v0 verification."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

import e3nn
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]

# Pinned SHA-256 digests and counts for the official `v0.1.0` TensorQA-v0 release.
OFFICIAL_V0_TARBALL_SHA256 = "8a51673f849c217febeebcbb934c40c2fab6dcc28884e689b99b6cd36c85c782"
OFFICIAL_V0_SPLIT_SHA256 = {
    "train": "1ac8fbd9da6135b55a6aba43a89a2c5e0cbb1d0765d92f492a6996288902c68f",
    "val": "0b0cb4baae8d31b2b0ec5a0d35931ed68414236b7b7646445bbcafff001ce7fd",
    "test": "24bf8adffdee1af126f914dce912c0c5d3a1a5143eea6f2bbdfe4a470791139b",
    "test_heldout": "5f223d8f46917d63f2eb3abb1280c82914a29990ea27042bc09de57bebcbeffe",
}
OFFICIAL_V0_SPLIT_COUNTS = {
    "train": 61619,
    "val": 7732,
    "test": 7703,
    "test_heldout": 7703,
}
OFFICIAL_V0_QUESTION_COUNT = 84757
OFFICIAL_V0_UNIQUE_STRINGS = 584
OFFICIAL_V0_MOLECULE_COUNT = 19843
# Canonical SHA-256 over the ordered `(split, task, atom, question)` stream (`id = 0..84756`).
OFFICIAL_V0_QUESTIONS_CANONICAL_SHA256 = "e771153eb1de961de74dc82b7a4cae2001280fb7e8aa60bf05a62a430dcb6c34"
# Exact file SHA-256 of `data/tensorqa_v0.json` produced by `scripts/06b_unpack_release.py`.
OFFICIAL_V0_UNPACKED_JSON_SHA256 = "76cf64ea621b5c5e521caeb1bbc6511bfb4b0fac7a521e79d67f5249a73630d3"


def file_sha256(path: str | os.PathLike | None) -> str | None:
    """Return the hex SHA-256 of `path`, or `None` when `path` is unset or missing."""
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str | None:
    """Return the current git commit hash, or `None` outside a git checkout."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def canonical_questions_sha256(questions: list[dict]) -> str:
    """Compute canonical SHA-256 over ordered `(split, task, atom, question)` tuples."""
    tuples = [
        (
            str(q["split"]),
            str(q["task"]),
            -1 if q["task"] == "dipole" else int(q["atom"] if "atom" in q else q["atom_index"]),
            str(q["question"]),
        )
        for q in questions
    ]
    payload = json.dumps(tuples, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_official_v0_questions(path: str | os.PathLike) -> dict:
    """Verify an official TensorQA-v0 `tensorqa_v0.json`, `tensorqa_v0.tar.gz`, or split directory.

    Raises `ValueError` if counts (`84,757` questions, `584` unique strings, per-split counts)
    or pinned SHA-256 digests do not match the `v0.1.0` release.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"TensorQA-v0 path not found: {target}")

    raw_file_sha = None
    if target.is_dir():
        rows = []
        for sp, expected_sha in OFFICIAL_V0_SPLIT_SHA256.items():
            sp_file = target / f"{sp}.jsonl"
            if not sp_file.exists():
                raise ValueError(f"Missing split file {sp_file}")
            got_sha = file_sha256(sp_file)
            if got_sha != expected_sha:
                raise ValueError(f"{sp}.jsonl SHA-256 mismatch: expected {expected_sha}, got {got_sha}")
            for line in sp_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        rows.sort(key=lambda r: int(r["id"]))
        questions = [
            {
                "split": r["split"],
                "task": r["task"],
                "atom": -1 if r["task"] == "dipole" else int(r["atom_index"]),
                "question": r["question"],
            }
            for r in rows
        ]
    elif target.name.endswith(".tar.gz") or target.suffix == ".tgz":
        raw_file_sha = file_sha256(target)
        if raw_file_sha != OFFICIAL_V0_TARBALL_SHA256:
            raise ValueError(
                f"Tarball SHA-256 mismatch: expected {OFFICIAL_V0_TARBALL_SHA256}, got {raw_file_sha}"
            )
        rows = []
        with tarfile.open(target, "r:gz") as tar:
            for sp, expected_sha in OFFICIAL_V0_SPLIT_SHA256.items():
                member = tar.extractfile(f"tensorqa_v0/{sp}.jsonl")
                if member is None:
                    raise ValueError(f"Missing tensorqa_v0/{sp}.jsonl inside {target}")
                raw = member.read()
                got_sha = hashlib.sha256(raw).hexdigest()
                if got_sha != expected_sha:
                    raise ValueError(f"{sp}.jsonl SHA-256 mismatch: expected {expected_sha}, got {got_sha}")
                for line in raw.decode("utf-8").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
        rows.sort(key=lambda r: int(r["id"]))
        questions = [
            {
                "split": r["split"],
                "task": r["task"],
                "atom": -1 if r["task"] == "dipole" else int(r["atom_index"]),
                "question": r["question"],
            }
            for r in rows
        ]
    else:
        raw_file_sha = file_sha256(target)
        questions = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(questions, list):
            raise ValueError(f"Expected JSON list of questions in {target}")

    n_total = len(questions)
    if n_total != OFFICIAL_V0_QUESTION_COUNT:
        raise ValueError(
            f"Question count mismatch: expected {OFFICIAL_V0_QUESTION_COUNT}, got {n_total} "
            "(a fresh rebuild is not the official v0 file)"
        )

    split_counts = dict(Counter(q["split"] for q in questions))
    if split_counts != OFFICIAL_V0_SPLIT_COUNTS:
        raise ValueError(
            f"Split counts mismatch: expected {OFFICIAL_V0_SPLIT_COUNTS}, got {split_counts}"
        )

    n_unique = len({q["question"] for q in questions})
    if n_unique != OFFICIAL_V0_UNIQUE_STRINGS:
        raise ValueError(
            f"Unique question string count mismatch: expected {OFFICIAL_V0_UNIQUE_STRINGS}, got {n_unique}"
        )

    canon_sha = canonical_questions_sha256(questions)
    if canon_sha != OFFICIAL_V0_QUESTIONS_CANONICAL_SHA256:
        raise ValueError(
            f"Official v0 question hash mismatch: expected {OFFICIAL_V0_QUESTIONS_CANONICAL_SHA256}, got {canon_sha}"
        )

    return {
        "path": str(target),
        "file_sha256": raw_file_sha,
        "canonical_questions_sha256": canon_sha,
        "n_questions": n_total,
        "n_unique_strings": n_unique,
        "split_counts": split_counts,
    }


def build_provenance(
    *,
    questions_path: str | os.PathLike | None,
    text_cache_path: str | os.PathLike | None = None,
    split_counts: dict[str, int],
    eval_rotations: int,
    eval_reflections: bool = False,
    eval_seed: int = 0,
) -> dict:
    """Build the `provenance` block saved in `results.json`."""
    return {
        "questions_path": str(questions_path) if questions_path else None,
        "questions_sha256": file_sha256(questions_path),
        "text_cache_path": str(text_cache_path) if text_cache_path else None,
        "text_cache_sha256": file_sha256(text_cache_path),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "eval_rotations": int(eval_rotations),
        "eval_reflections": bool(eval_reflections),
        "eval_seed": int(eval_seed),
        "git_commit": current_git_commit(),
        "torch_version": torch.__version__,
        "e3nn_version": e3nn.__version__,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify official TensorQA-v0 question file or tarball.")
    parser.add_argument("--check-v0", required=True, help="Path to tensorqa_v0.json, tensorqa_v0.tar.gz, or split dir.")
    args = parser.parse_args(argv)
    info = verify_official_v0_questions(args.check_v0)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

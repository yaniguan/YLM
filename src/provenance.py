"""Record data hashes, split counts, eval settings, and package versions for `results.json`."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

import e3nn
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


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

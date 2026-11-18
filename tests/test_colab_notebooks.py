"""Verify committed Colab notebooks clone a git branch that actually exists."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_DIR = REPO_ROOT / "colab"


def extract_notebook_default_branch(nb_path: Path) -> str:
    """Return the default `BRANCH` string configured in a Colab `.ipynb` file."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell.get("source", []))
        tree = ast.parse(code, filename=str(nb_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "BRANCH" for t in node.targets):
                continue
            val = node.value
            # Direct string literal: BRANCH = 'main'
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                return val.value
            # Env override with default: BRANCH = os.environ.get('YLM_BRANCH', 'main')
            if (
                isinstance(val, ast.Call)
                and len(val.args) >= 2
                and isinstance(val.args[1], ast.Constant)
                and isinstance(val.args[1].value, str)
            ):
                return val.args[1].value
    raise AssertionError(f"No default BRANCH assignment found in {nb_path.name}")


def branch_exists_in_repo(branch: str, repo_root: Path = REPO_ROOT) -> bool:
    """Check local refs (`refs/heads/<branch>`, `refs/remotes/origin/<branch>`) and fall back to `git ls-remote`."""
    for ref in (f"refs/heads/{branch}", f"refs/remotes/origin/{branch}"):
        proc = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", ref],
            cwd=repo_root,
            check=False,
        )
        if proc.returncode == 0:
            return True
    remote = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", "origin", branch],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    return remote.returncode == 0


def assert_notebook_branch_exists(nb_path: Path, repo_root: Path = REPO_ROOT) -> str:
    branch = extract_notebook_default_branch(nb_path)
    assert branch_exists_in_repo(branch, repo_root=repo_root), (
        f"{nb_path.name} checks out missing branch {branch!r}"
    )
    return branch


def test_all_colab_notebooks_clone_existing_branch():
    notebooks = sorted(COLAB_DIR.glob("*.ipynb"))
    assert len(notebooks) >= 4
    for nb_path in notebooks:
        branch = assert_notebook_branch_exists(nb_path)
        assert branch == "main"


def test_missing_colab_branch_is_rejected(tmp_path):
    bad_nb = tmp_path / "stale.ipynb"
    bad_nb.write_text(
        json.dumps({
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {},
                    "outputs": [],
                    "source": ["BRANCH = 'feat/llm-irreps-ablation-oct30'\n"],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="missing branch"):
        assert_notebook_branch_exists(bad_nb)

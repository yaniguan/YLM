"""Tests for official-v0 multi-seed Option A/B/C and data-fraction table/plot builders."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from src.provenance import (
    OFFICIAL_V0_SPLIT_COUNTS,
    OFFICIAL_V0_UNPACKED_JSON_SHA256,
    REPO_ROOT,
)


def _load_script(rel_path: str, mod_name: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_run(
    runs_dir: Path,
    name: str,
    *,
    text: str = "minilm",
    option: str = "A",
    seed: int = 0,
    train_fraction: float = 1.0,
    attn_sup: float = 0.0,
    questions_sha256: str = OFFICIAL_V0_UNPACKED_JSON_SHA256,
    eval_rotations: int = 32,
    eval_reflections: bool = False,
    force_mae: float = 0.0070,
    force_cos: float = 0.950,
    atom_acc: float = 0.800,
):
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": {
            "text": text,
            "option": option,
            "name": name,
            "epochs": 60 if train_fraction == 1.0 else 150,
            "batch_size": 128,
            "lr": 0.001,
            "train_fraction": train_fraction,
            "eval_rotations": eval_rotations,
            "eval_reflections": eval_reflections,
            "attn_sup": attn_sup,
            "seed": seed,
        },
        "provenance": {
            "questions_path": "data/tensorqa_v0.json",
            "questions_sha256": questions_sha256,
            "text_cache_path": f"data/text_{text}.pt",
            "text_cache_sha256": "a" * 64,
            "split_counts": dict(OFFICIAL_V0_SPLIT_COUNTS),
            "eval_rotations": eval_rotations,
            "eval_reflections": eval_reflections,
            "eval_seed": 0,
            "git_commit": "deadbeef",
            "torch_version": "2.4.0",
            "e3nn_version": "0.6.0",
        },
        "scale": [0.81, 0.06],
        "test": {
            "dipole": {
                "mae": 0.080 + 0.001 * seed,
                "cosine": 0.980 - 0.001 * seed,
                "eq_err_mean": 2e-6,
                "eq_err_max": 1e-4,
            },
            "force": {
                "mae": force_mae + 0.0002 * seed,
                "cosine": force_cos - 0.002 * seed,
                "eq_err_mean": 3e-6,
                "eq_err_max": 4e-4,
                "atom_acc": atom_acc - 0.01 * seed,
            },
        },
        "test_heldout": {
            "dipole": {
                "mae": 0.250 + 0.002 * seed,
                "cosine": 0.810 - 0.002 * seed,
                "eq_err_mean": 2e-6,
                "eq_err_max": 1e-4,
            },
            "force": {
                "mae": force_mae + 0.004 + 0.0002 * seed,
                "cosine": force_cos - 0.05 - 0.002 * seed,
                "eq_err_mean": 3e-6,
                "eq_err_max": 4e-4,
                "atom_acc": atom_acc - 0.05 - 0.01 * seed,
            },
        },
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_17_make_options_table_enforces_provenance_and_separates_b_variants(tmp_path: Path):
    mod17 = _load_script("scripts/17_make_options_table.py", "make_options_table_17")
    runs = tmp_path / "runs"
    for seed in (0, 1, 2):
        _write_run(runs, f"v0_optA_minilm_s{seed}", option="A", seed=seed, force_mae=0.0072, atom_acc=0.76)
        _write_run(runs, f"v0_optB_minilm_s{seed}", option="B", seed=seed, force_mae=0.0068, atom_acc=0.70)
        _write_run(
            runs,
            f"v0_optB_attn0p1_minilm_s{seed}",
            option="B",
            attn_sup=0.1,
            seed=seed,
            force_mae=0.0067,
            atom_acc=0.84,
        )
        _write_run(runs, f"v0_optC_minilm_s{seed}", option="C", seed=seed, force_mae=0.0050, atom_acc=0.91)

    out_md = tmp_path / "results_options_v0.md"
    out_json = tmp_path / "results_options_v0.json"
    md, summary = mod17.build_options_table([str(runs)], out_md=out_md, out_json=out_json, min_seeds=3)

    assert "Option A | minilm | 3 |" in md
    assert "Option B | minilm | 3 |" in md
    assert "Option B (`--attn_sup 0.1`) | minilm | 3 |" in md
    assert "Option C | minilm | 3 |" in md
    assert "±" in md
    assert [r["row_id"] for r in summary["rows"]] == ["A", "B", "B_attn_sup", "C"]

    # Wrong question file hash (e.g. fresh rebuild) must be rejected
    bad_hash = tmp_path / "bad_hash"
    for seed in (0, 1, 2):
        _write_run(bad_hash, f"optA_s{seed}", option="A", seed=seed, questions_sha256="f" * 64)
    with pytest.raises(ValueError, match="questions_sha256"):
        mod17.build_options_table([str(bad_hash)], out_md=tmp_path / "x.md", min_seeds=3)

    # 4-rotation + reflection Colab protocol must be rejected from the v0 32-rotation table
    bad_rot = tmp_path / "bad_rot"
    for seed in (0, 1, 2):
        _write_run(bad_rot, f"optA_s{seed}", option="A", seed=seed, eval_rotations=4, eval_reflections=True)
    with pytest.raises(ValueError, match="eval_rotations"):
        mod17.build_options_table([str(bad_rot)], out_md=tmp_path / "x.md", min_seeds=3)

    # Fewer than 3 seeds must be rejected
    too_few = tmp_path / "too_few"
    for seed in (0, 1):
        _write_run(too_few, f"optB_s{seed}", option="B", seed=seed)
    with pytest.raises(ValueError, match="at least 3 seeds"):
        mod17.build_options_table([str(too_few)], out_md=tmp_path / "x.md", min_seeds=3)


def test_07_make_table_and_08_analyze_handle_multiseed_fractions_and_ignore_b_c(tmp_path: Path):
    mod07 = _load_script("scripts/07_make_table.py", "make_table_07")
    mod08 = _load_script("scripts/08_analyze.py", "analyze_08")

    # Existing runs_slurm reproduces committed results.md
    baseline_md = tmp_path / "baseline_results.md"
    got_baseline = mod07.build_table([str(REPO_ROOT / "runs_slurm")], out_path=str(baseline_md))
    expected_baseline = (REPO_ROOT / "results.md").read_text(encoding="utf-8").strip()
    assert got_baseline.strip() == expected_baseline

    # Add seeds 1 and 2 for 3%/10%/30% plus Option B/C runs in a shared folder; B/C must not pollute Option A rows
    extra = tmp_path / "extra_runs"
    for seed in (1, 2):
        for frac in (0.03, 0.1, 0.3):
            for text in ("minilm", "matscibert", "taskid"):
                _write_run(extra, f"frac_{text}_f{frac}_s{seed}", text=text, option="A", seed=seed, train_fraction=frac)
        _write_run(extra, f"v0_optB_minilm_s{seed}", text="minilm", option="B", seed=seed)
        _write_run(extra, f"v0_optC_minilm_s{seed}", text="minilm", option="C", seed=seed)

    combined_md = tmp_path / "combined_results.md"
    text_md = mod07.build_table([str(REPO_ROOT / "runs_slurm"), str(extra)], out_path=str(combined_md))
    # 100% MiniLM stays at 5 seeds (not 7 or 9 from B/C!), and 3%/10%/30% all become 3 seeds with ± std
    assert "| YLM (Option A) + MiniLM | 100% | 5 |" in text_md
    for label in ("YLM (Option A) + MiniLM", "YLM (Option A) + MatSciBERT", "Baseline v: task ID instead of text"):
        for pct in ("30%", "10%", "3%"):
            matching = [ln for ln in text_md.splitlines() if f"| {label} | {pct} | 3 |" in ln]
            assert len(matching) == 2  # test and test_heldout
            assert "±" in matching[0]

    plot_dir = tmp_path / "analysis_out"
    plot_dir.mkdir()
    mod08.sample_efficiency([str(REPO_ROOT / "runs_slurm"), str(extra)], str(plot_dir))
    assert (plot_dir / "sample_efficiency.png").exists()
    assert (plot_dir / "sample_efficiency.png").stat().st_size > 5000


def test_ylm_train_notebook_syntax_and_v0_defaults():
    nb_path = REPO_ROOT / "colab" / "ylm_train.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) >= 4
    for src in code_cells:
        ast.parse(src)
    joined = "\n".join(code_cells)
    assert "06b_unpack_release.py" in joined
    assert "17_make_options_table.py" in joined
    assert "os.environ.get('YLM_BRANCH', 'main')" in joined
    assert "'32' if FULL else '4'" in joined

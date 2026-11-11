"""Prepare multi-domain Universe Foundation-Model JSONL splits (train/val/test/test_heldout).

Examples:
  # 1) Zero-download multi-domain CPU smoke dataset (drugs + electrolytes + PBC crystals + reactive TS)
  python scripts/31_prepare_universe_data.py --smoke --out data/universe_smoke

  # 2) Convert real DFT .extxyz / .xyz / .cif / .sdf files (SPICE-v2, OMol25, MPtrj, Transition1x)
  python scripts/31_prepare_universe_data.py --inputs path/to/spice.extxyz --lot wb97m --domain spice --out data/universe_spice
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.universe_data import (  # noqa: E402
    build_cpu_smoke_universe_dataset,
    prepare_universe_from_files,
    save_universe_splits,
)


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Prepare Universe multi-task JSONL dataset splits.")
    parser.add_argument("--inputs", nargs="*", default=[], help="Input .xyz/.extxyz/.sdf/.cif/.poscar files.")
    parser.add_argument("--smoke", action="store_true", help="Generate zero-download multi-domain CPU smoke dataset.")
    parser.add_argument("--n_conformers", type=int, default=12, help="Conformers per archetype in --smoke mode.")
    parser.add_argument("--lot", type=str, default="wb97m", help="Default Level-of-Theory tag for --inputs.")
    parser.add_argument("--domain", type=str, default="custom", help="Domain tag for --inputs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/universe_smoke", help="Output directory for JSONL splits.")
    args = parser.parse_args(argv)

    if args.smoke or not args.inputs:
        splits = build_cpu_smoke_universe_dataset(
            n_conformers_per_archetype=args.n_conformers,
            seed=args.seed,
        )
    else:
        splits = prepare_universe_from_files(
            args.inputs,
            lot=args.lot,
            domain=args.domain,
            seed=args.seed,
        )

    saved_paths = save_universe_splits(splits, args.out)
    all_recs = [r for recs in splits.values() for r in recs]
    manifest = {
        "out_dir": str(Path(args.out)),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "domain_counts": dict(Counter(r.domain for r in all_recs)),
        "lot_counts": dict(Counter(r.lot for r in all_recs)),
        "subtask_counts": dict(Counter(r.subtask for r in all_recs)),
        "files": saved_paths,
    }
    manifest_path = Path(args.out) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()

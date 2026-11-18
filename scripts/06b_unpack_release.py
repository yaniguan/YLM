"""Unpack the official `v0.1.0` `tensorqa_v0.tar.gz` release into `data/qm9_xtb.pkl` and `data/tensorqa_v0.json`.

Using this script avoids rebuilding labels with `tblite` / `RDKit` (which produces a slightly different
question count across builds) and verifies the pinned v0 SHA-256 digests and counts (`84,757` / `584`).

Run:
  python scripts/06b_unpack_release.py --tarball data/release/tensorqa_v0.tar.gz
  python scripts/06b_unpack_release.py --download
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.provenance import (  # noqa: E402
    OFFICIAL_V0_MOLECULE_COUNT,
    OFFICIAL_V0_UNPACKED_JSON_SHA256,
    file_sha256,
    verify_official_v0_questions,
)

RELEASE_TARBALL_URL = "https://github.com/yaniguan/YLM/releases/download/v0.1.0/tensorqa_v0.tar.gz"


def unpack_v0_tarball(
    tarball_path: str | os.PathLike,
    *,
    labels_out: str | os.PathLike = "data/qm9_xtb.pkl",
    questions_out: str | os.PathLike = "data/tensorqa_v0.json",
) -> dict:
    tar_p = Path(tarball_path)
    verify_official_v0_questions(tar_p)

    rows = []
    with tarfile.open(tar_p, "r:gz") as tar:
        for sp in ("train", "val", "test", "test_heldout"):
            member = tar.extractfile(f"tensorqa_v0/{sp}.jsonl")
            assert member is not None
            for line in member.read().decode("utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))

    rows.sort(key=lambda r: int(r["id"]))
    mol_by_qm9: dict[int, int] = {}
    labels: list[dict] = []
    questions: list[dict] = []

    for r in rows:
        qidx = int(r["qm9_index"])
        if qidx not in mol_by_qm9:
            mol_by_qm9[qidx] = len(labels)
            n_at = len(r["atomic_numbers"])
            labels.append({
                "qm9_index": qidx,
                "numbers": [int(z) for z in r["atomic_numbers"]],
                "pos": r["positions_bohr"],
                "dipole": [0.0, 0.0, 0.0],
                "forces": [[0.0, 0.0, 0.0] for _ in range(n_at)],
            })
        m = mol_by_qm9[qidx]
        if r["task"] == "dipole":
            labels[m]["dipole"] = r["answer"]
            atom = -1
        else:
            atom = int(r["atom_index"])
            labels[m]["forces"][atom] = r["answer"]
        questions.append({
            "mol": m,
            "split": r["split"],
            "task": r["task"],
            "atom": atom,
            "question": r["question"],
        })

    assert len(labels) == OFFICIAL_V0_MOLECULE_COUNT

    labels_p = Path(labels_out)
    questions_p = Path(questions_out)
    labels_p.parent.mkdir(parents=True, exist_ok=True)
    questions_p.parent.mkdir(parents=True, exist_ok=True)

    with open(labels_p, "wb") as handle:
        pickle.dump(labels, handle)
    questions_p.write_bytes((json.dumps(questions, indent=0) + "\n").encode("utf-8"))

    info = verify_official_v0_questions(questions_p)
    assert file_sha256(questions_p) == OFFICIAL_V0_UNPACKED_JSON_SHA256
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unpack official TensorQA-v0 release tarball.")
    parser.add_argument("--tarball", default="data/release/tensorqa_v0.tar.gz")
    parser.add_argument("--download", action="store_true", help="Download tarball from v0.1.0 release if missing.")
    parser.add_argument("--labels-out", default="data/qm9_xtb.pkl")
    parser.add_argument("--questions-out", default="data/tensorqa_v0.json")
    args = parser.parse_args(argv)

    tar_p = Path(args.tarball)
    if args.download and not tar_p.exists():
        tar_p.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {RELEASE_TARBALL_URL} -> {tar_p}")
        urllib.request.urlretrieve(RELEASE_TARBALL_URL, tar_p)

    info = unpack_v0_tarball(tar_p, labels_out=args.labels_out, questions_out=args.questions_out)
    print(f"Unpacked {info['n_questions']} official v0 questions ({info['n_unique_strings']} unique strings).")
    print(f"questions_sha256: {info['file_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

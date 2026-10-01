"""Export TensorQA-v0 as plain JSONL files for release.

Each line is one question and has everything needed (no pickle, no code from this repo):
  {"id", "split", "task", "question", "atom_index", "atomic_numbers", "positions_bohr",
   "answer", "answer_unit", "qm9_index"}

Output:
  data/release/tensorqa_v0/{train,val,test,test_heldout}.jsonl
  data/release/tensorqa_v0/DATA_CARD.md
  data/release/tensorqa_v0.tar.gz

Run:
  python scripts/06_export_release.py
"""
import json
import os
import pickle
import shutil
import tarfile

UNIT = {"dipole": "e*bohr", "force": "hartree/bohr"}


def main():
    labels = pickle.load(open("data/qm9_xtb.pkl", "rb"))
    questions = json.load(open("data/tensorqa_v0.json"))
    out = "data/release/tensorqa_v0"
    os.makedirs(out, exist_ok=True)

    files = {s: open(f"{out}/{s}.jsonl", "w") for s in ["train", "val", "test", "test_heldout"]}
    for k, q in enumerate(questions):
        lab = labels[q["mol"]]
        answer = lab["dipole"] if q["task"] == "dipole" else lab["forces"][q["atom"]]
        row = {
            "id": k,
            "split": q["split"],
            "task": q["task"],
            "question": q["question"],
            "atom_index": q["atom"] if q["task"] == "force" else None,
            "atomic_numbers": [int(z) for z in lab["numbers"]],
            "positions_bohr": [[round(float(x), 6) for x in p] for p in lab["pos"]],
            "answer": [round(float(x), 8) for x in answer],
            "answer_unit": UNIT[q["task"]],
            "qm9_index": int(lab["qm9_index"]),
        }
        files[q["split"]].write(json.dumps(row) + "\n")
    for f in files.values():
        f.close()

    shutil.copy("DATA_CARD.md", f"{out}/DATA_CARD.md")
    with tarfile.open("data/release/tensorqa_v0.tar.gz", "w:gz") as tar:
        tar.add(out, arcname="tensorqa_v0")
    for s in files:
        n = sum(1 for _ in open(f"{out}/{s}.jsonl"))
        print(f"{s}: {n} questions")
    print("wrote data/release/tensorqa_v0.tar.gz",
          round(os.path.getsize("data/release/tensorqa_v0.tar.gz") / 1e6, 1), "MB")


if __name__ == "__main__":
    main()

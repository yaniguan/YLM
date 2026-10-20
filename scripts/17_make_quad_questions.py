"""TensorQA-quad: l=2 questions. One quantity (traceless molecular quadrupole, HF/6-31G*, origin = charge centre),
asked with 8 wordings; the last two are held out (test_heldout). Splits follow tensorqa_v0.json (by molecule).

  python scripts/17_make_quad_questions.py     # merges data/quad_part*.pkl -> data/qm9_quad.pkl, data/tensorqa_quad.json
"""
import glob
import json
import pickle

TEMPLATES = [
    "What is the quadrupole moment of the molecule?",
    "Give the traceless quadrupole tensor of this molecule.",
    "Report the molecular quadrupole.",
    "How is the charge distribution of the molecule distributed in second moments? Give the quadrupole.",
    "Compute the second-order multipole moment of the molecule.",
    "What is the electric quadrupole of the system?",
    "Tell me the quadrupole of this structure.",           # held out
    "State the traceless second moment of the charge density.",  # held out
]
HELDOUT = 2


def main():
    quad = {}
    for f in sorted(glob.glob("data/quad_part*.pkl")):
        quad.update(pickle.load(open(f, "rb")))
    pickle.dump(quad, open("data/qm9_quad.pkl", "wb"))
    split_of = {q["mol"]: q["split"] for q in json.load(open("data/tensorqa_v0.json")) if q["split"] != "test_heldout"}
    out, k = [], 0
    for m, Q in sorted(quad.items()):
        if m not in split_of:
            continue
        sp = split_of[m]
        train_t = TEMPLATES[:-HELDOUT]
        out.append(dict(mol=m, split=sp, task="quad", target=Q.tolist(), question=train_t[k % len(train_t)]))
        if sp == "test":
            for t in TEMPLATES[-HELDOUT:]:
                out.append(dict(mol=m, split="test_heldout", task="quad", target=Q.tolist(), question=t))
        k += 1
    json.dump(out, open("data/tensorqa_quad.json", "w"))
    from collections import Counter
    print(len(quad), "labelled molecules;", Counter(q["split"] for q in out))


if __name__ == "__main__":
    main()

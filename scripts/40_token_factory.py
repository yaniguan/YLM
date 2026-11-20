"""Token factory v1 (issue #80): propose, verify, dedup and label new force questions on train molecules.

Arms: A = 02's templates on every describable atom, C = mutations (verifier test, never kept),
B = a teacher LM (only with --teacher). See src/token_factory.py for the verifiers.

Outputs in --out:
  kept.jsonl      accepted items with the xTB force (hartree/bohr) as force_hartree_per_bohr
  rejected.jsonl  rejected items with stage and reason (all arm C items end here)
  stats.json      per-arm stage counts, reasons, yield, coverage, timings, input hashes, official_v0

--questions must be the official TensorQA-v0 file (its SHA-256 is compared to the pinned one).
Any other file stops the run, unless --allow_unofficial is given; stats.json then has official_v0: false.

Run:
  python scripts/40_token_factory.py --n_mols 2000
  python scripts/40_token_factory.py --n_mols 200 --teacher Qwen/Qwen2.5-0.5B-Instruct --teacher_n 200
"""
import argparse
import json
import os
import pickle
import platform
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import rdkit
from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.token_factory import (  # noqa: E402
    OFFICIAL_V0_UNPACKED_JSON_SHA256,
    VERIFIER_VERSION,
    build_teacher_prompt,
    file_sha256,
    mutate,
    normalise,
    parse_teacher_output,
    propose_templates,
    run_factory,
)

RDLogger.DisableLog("rdApp.*")

LICENCES = {
    "qm9_gdb9_sdf": "QM9 (Ramakrishnan et al. 2014), open terms; cite Sci. Data 1, 140022",
    "xtb_labels": "TensorQA-v0 labels, CC BY 4.0 (see DATA_CARD.md)",
}


def run_teacher(model_id, prompts, batch_size, seed):
    """Sample one answer per prompt on CPU. -> (texts, info)"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).eval()
    texts, n_tokens = [], 0
    t0 = time.perf_counter()
    for s in range(0, len(prompts), batch_size):
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                 for p in prompts[s:s + batch_size]]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=0.7, max_new_tokens=96,
                                 pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        n_tokens += int((new != tok.pad_token_id).sum())
        texts += tok.batch_decode(new, skip_special_tokens=True)
    seconds = time.perf_counter() - t0
    import transformers
    info = dict(model_id=model_id, revision=getattr(model.config, "_commit_hash", None),
                transformers_version=transformers.__version__, torch_version=torch.__version__,
                tokens_generated=n_tokens, seconds=round(seconds, 2),
                tokens_per_second=round(n_tokens / max(seconds, 1e-9), 2))
    return texts, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/tensorqa_v0.json")
    parser.add_argument("--labels", default="data/qm9_xtb.pkl")
    parser.add_argument("--sdf", default="data/gdb9.sdf")
    parser.add_argument("--n_mols", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher", default=None, help="HF model id; arm B runs only if given")
    parser.add_argument("--teacher_n", type=int, default=200)
    parser.add_argument("--teacher_batch", type=int, default=8)
    parser.add_argument("--out", default="runs/token_factory")
    parser.add_argument("--allow_unofficial", action="store_true",
                        help="run on a question file that is not the official v0 file (official_v0: false)")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    timings = {}

    # the train split comes from --questions, so it must be the pinned official file
    questions_sha = file_sha256(args.questions)
    official_v0 = questions_sha == OFFICIAL_V0_UNPACKED_JSON_SHA256
    if not official_v0:
        msg = (f"{args.questions} is not the official TensorQA-v0 file: SHA-256 {questions_sha}, "
               f"expected {OFFICIAL_V0_UNPACKED_JSON_SHA256}")
        if not args.allow_unofficial:
            sys.exit(msg + ". Pass --allow_unofficial to run anyway.")
        print("WARNING: " + msg + " (--allow_unofficial, stats.json gets official_v0: false)")

    t0 = time.perf_counter()
    questions = json.load(open(args.questions))
    train_ids = sorted({q["mol"] for q in questions if q["split"] == "train"})
    existing_pairs = {(q["mol"], q["atom"]) for q in questions if q["task"] == "force"}
    existing_questions = {(q["mol"], q["atom"], normalise(q["question"])) for q in questions if q["task"] == "force"}
    labels = pickle.load(open(args.labels, "rb"))
    supplier = Chem.SDMolSupplier(args.sdf, removeHs=False, sanitize=True)
    picked = sorted(rng.sample(train_ids, min(args.n_mols, len(train_ids))))
    mols = {}
    for m in picked:
        mol = supplier[labels[m]["qm9_index"]]
        assert [a.GetAtomicNum() for a in mol.GetAtoms()] == list(labels[m]["numbers"])
        mols[m] = mol
    timings["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    items_a = [it for m in picked for it in propose_templates(m, mols[m])]
    timings["propose_A"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    by_mol = defaultdict(list)
    for it in items_a:
        by_mol[it["mol"]].append(it)
    items_c = []
    for m in picked:  # one random base item per molecule -> one variant per mutation kind
        if by_mol[m]:
            items_c += mutate(rng.choice(by_mol[m]), mols[m], rng)
    timings["propose_C"] = time.perf_counter() - t0

    items_b, teacher_info = [], None
    if args.teacher:
        t0 = time.perf_counter()
        teacher_mols = [rng.choice(picked) for _ in range(args.teacher_n)]
        texts, teacher_info = run_teacher(args.teacher, [build_teacher_prompt(mols[m]) for m in teacher_mols],
                                          args.teacher_batch, args.seed)
        for m, text in zip(teacher_mols, texts):
            # unparsable output stays an arm-B proposal and fails V0
            items_b.append(parse_teacher_output(text, mol=m) or dict(mol=m, arm="B", raw=text))
        teacher_info["unparsed"] = sum("question" not in it for it in items_b)
        timings["propose_B"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    kept, rejected, stats = run_factory(items_a + items_b + items_c, mols, labels, set(train_ids), existing_pairs,
                                       existing_questions)
    timings["verify_total"] = time.perf_counter() - t0

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "kept.jsonl"), "w") as f:
        for it in kept:
            f.write(json.dumps(it) + "\n")
    with open(os.path.join(args.out, "rejected.jsonl"), "w") as f:
        for it in rejected:
            f.write(json.dumps(it) + "\n")
    stats.update(
        inputs={name: dict(path=path, sha256=file_sha256(path))
                for name, path in (("questions", args.questions), ("labels", args.labels), ("sdf", args.sdf))},
        official_v0=official_v0,
        seed=args.seed, n_mols=len(picked), teacher=teacher_info, verifier_version=VERIFIER_VERSION,
        versions=dict(python=platform.python_version(), rdkit=rdkit.__version__, numpy=np.__version__),
        licences=LICENCES, seconds_per_step={k: round(v, 3) for k, v in timings.items()},
    )
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w"), indent=1)

    print(f"molecules: {len(picked)}  out: {args.out}  official_v0: {official_v0}")
    for arm in sorted(stats["proposals"]):
        print(f"arm {arm}: proposals {stats['proposals'][arm]}  kept {stats['kept'].get(arm, 0)}  "
              f"v0 copies removed {stats['dedup_v0_removed'].get(arm, 0)}  "
              f"yield/1k {stats['yield_per_1k'][arm]}  new pairs {stats['new_mol_atom_pairs'].get(arm, 0)}  "
              f"V2c/V2d disagree {stats['v2c_v2d_disagreements'].get(arm, 0)}")
        print("  passed after stage:", stats["passed_after_stage"][arm])
        print("  top reasons:", Counter(stats["rejection_reasons"].get(arm, {})).most_common(3))
    for kind, c in stats["mutations"].items():
        print(f"  mutation {kind:12s} {c}")
    print("dedup removed:", stats["dedup_removed"], " dedup_v0 removed:", stats["dedup_v0_removed"],
          " constraints:", stats["constraint_histogram"])
    print("seconds:", stats["seconds_per_step"])


if __name__ == "__main__":
    main()

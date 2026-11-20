# Token factory v1: force questions

Issue #80. The factory makes new training items, checks them with exact verifiers, and keeps only the ones that pass.

## What an item is

A force question about one atom of a QM9 `train` molecule, for example "What is the force on the carbon atom bonded to three hydrogens and one carbon?". The answer is the xTB force on that atom (hartree/bohr), looked up from the labels. Nothing in the answer is generated.

## Proposers

- **Arm A**: every atom that `scripts/02_make_questions.py` can describe uniquely, times the 4 train force frames. No language model.
- **Arm B**: an optional small teacher LM writes the question and names the atom (`--teacher`). Off by default.
- **Arm C**: mutated items (negation, "a nitrogen", "or", two atoms, plural, a partial neighbour list, a wrong atom, held-out frames, extra words, uppercase, reordered neighbours). These are a regression test of the hand-written mutation kinds. They must all be rejected and are never kept. They are not evidence that the verifiers are complete.

## Verifiers (all exact, all on CPU, run in order)

| Stage | Check |
|---|---|
| V0 | format: keys, types, length |
| V1 | the molecule is in the `train` split |
| V2a | the frame is a train frame, and the description matches 02's closed grammar with the same letter case |
| V2b | exactly one "the <element> atom" phrase and at most one bonded clause (a backstop; V2a already enforces this) |
| V2c | `resolve()` picks exactly the claimed atom |
| V2d | the "at least" reading picks the same single atom |
| V3 | the xTB force on the atom exists and is finite |

## Official question file

`scripts/40_token_factory.py` compares the question-file sha256 to the pinned official v0 hash and stops on a mismatch, unless `--allow_unofficial` is set. In that case `stats.json` records `official_v0: false`.

Exact copies of existing TensorQA-v0 train questions are rejected at stage `dedup_v0` and reported as `dedup_v0_removed`. `kept_new_strings` is the number of kept items that are not those copies.

## Data card (provisional; unofficial question file)

The counts below come from `python scripts/40_token_factory.py --n_mols 2000` (arms A and C, seed 0) on a local copy of `data/tensorqa_v0.json`. That file's hash differs from the pinned official v0 hash in `src/provenance.py`. There is no committed `stats.json` for an official-file run. Do not use these items for training until the factory is re-run on the official file. The counts, including the 5,649 new-pair figure, may change.

| | Arm A | Arm C |
|---|---|---|
| proposals | 49,292 | 17,464 |
| kept | 42,840 (86.9%) | 0 |
| rejected | 6,452, all at V2d | 17,464 |

- Arm A: every rejection is V2d. These are 02's own template wordings that would also fit a second atom under the "at least" reading. The 42,840 kept figure includes exact copies of existing v0 train questions; the current code removes those copies at `dedup_v0`, so a re-run will report a lower kept count.
- Arm C (this run): every mutation kind then in the script was caught by the verifier we expected (V2a for 7 kinds, V2c for "partial" and "wrong atom"), 100% of the time. Uppercase and neighbour-reorder mutations were added later and were not part of this run.
- 5,649 kept (molecule, atom) pairs are not in TensorQA-v0.
- The earlier "11.4 s on one CPU core" timing is not kept: there is no committed result file for it.

## Limits

- No training run here, and no claim that these items improve YLM.
- Arm B (teacher LM) was not measured.
- Re-run on the official question file before using the items for training.

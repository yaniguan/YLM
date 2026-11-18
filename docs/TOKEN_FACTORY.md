# Token factory v1: force questions

Issue #80. The factory makes new training items, checks them with exact verifiers, and keeps only the ones that pass.

## What an item is

A force question about one atom of a QM9 `train` molecule, for example "What is the force on the carbon atom bonded to three hydrogens and one carbon?". The answer is the xTB force on that atom (hartree/bohr), looked up from the labels. Nothing in the answer is generated.

## Proposers

- **Arm A**: every atom that `scripts/02_make_questions.py` can describe uniquely, times the 4 train force frames. No language model.
- **Arm B**: an optional small teacher LM writes the question and names the atom (`--teacher`). Off by default.
- **Arm C**: mutated items (negation, "a nitrogen", "or", two atoms, plural, a partial neighbour list, a wrong atom, held-out frames, extra words). These test the verifiers. They must all be rejected and are never kept.

## Verifiers (all exact, all on CPU, run in order)

| Stage | Check |
|---|---|
| V0 | format: keys, types, length |
| V1 | the molecule is in the `train` split |
| V2a | the frame is a train frame, and the description is in 02's closed grammar |
| V2b | exactly one "the <element> atom" phrase and at most one bonded clause |
| V2c | `resolve()` picks exactly the claimed atom |
| V2d | the "at least" reading picks the same single atom |
| V3 | the xTB force on the atom exists and is finite |

## Data card (measured, CPU)

Run: `python scripts/40_token_factory.py --n_mols 2000` (arms A and C, seed 0).

| | Arm A | Arm C |
|---|---|---|
| proposals | 49,292 | 17,464 |
| kept | 42,840 (86.9%) | 0 |
| rejected | 6,452, all at V2d | 17,464 |

- Arm A: every rejection is V2d. These are 02's own template wordings that would also fit a second atom under the "at least" reading.
- Arm C: every mutation kind was caught by the verifier we expected (V2a for 7 kinds, V2c for "partial" and "wrong atom"), 100% of the time.
- 5,649 kept (molecule, atom) pairs are not in TensorQA-v0.
- Verifying all 66,756 items took 11.4 s on one CPU core.

## Limits

- No training run here, and no claim that these items improve YLM.
- Arm B (teacher LM) was not measured.
- The run above used a local copy of `data/tensorqa_v0.json` for the train split. Its hash differs from the pinned official v0 hash in `src/provenance.py`. Re-run on the official file before using the items for training.

# v0 highlights (evidence for a paper draft)

Each line is a claim, the number behind it, and where it comes from. Lowest-bar claims only; v0 is a feasibility result.

| # | Claim | Evidence | Source |
|---|---|---|---|
| 1 | Words can select a direction. | Force MAE 0.0064 vs 0.0424 (6.6x lower), cosine 0.96 vs 0.11, text vs task-ID, 5 vs 4 seeds | `results.md` |
| 2 | Exact equivariance without augmentation. | Mean rotation error ~2e-6 (float32) over 32 rotations/question; float64 test < 1e-9 | `results.md`, `tests/` |
| 3 | Works on unseen wordings with a science text encoder. | Held-out wordings: force cos 0.917, dipole cos 0.941 (MatSciBERT) vs 0.859 / 0.726 (MiniLM) | `results.md` |
| 4 | Attention is mostly interpretable. | Peak on asked atom or a bonded neighbour 94-97%; exact-atom accuracy is seed-noisy (0.08-0.94) so not a headline | `analysis/breakdown.md` |
| 5 | A reusable benchmark. | TensorQA-v0: 84,757 questions, 20k QM9 molecules, held-out templates, rotation protocol | `DATA_CARD.md` |
| 6 | Honest negative. | On the dipole (no atom selection needed) task ID is slightly better: 0.067 vs 0.082 MAE | `results.md` |

## Not yet supported (do not claim)

- Beating "coordinates as text into an LLM": that baseline did not finish.
- Anything about tensors (l=2), crystals, chirality, or generation.
- That a "parse text -> atom index -> force model" pipeline is worse (see Limitations in README).
- Real-DFT accuracy: labels are GFN2-xTB.

## Closest related work

EquiLLM (invariants only into the LLM) and Mol-StrucTok, KDD '26 (invariant discrete tokens for generation).
Both drop l>=1 direction before the language side; see `02_novelty_check.md`.

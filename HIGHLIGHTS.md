# v0 highlights (evidence for a paper draft)

Each line is a claim, the number behind it, and where it comes from. Lowest-bar claims only; v0 is a feasibility result.
Updated 2026-10-18 after the post-v0 baselines ([`results_baselines.md`](results_baselines.md)).

| # | Claim | Evidence | Source |
|---|---|---|---|
| 1 | Words can specify a direction in a model that stays exactly equivariant. | Force MAE 0.0064 vs 0.0424 for task ID (6.6x), cosine 0.96 vs 0.11 | `results.md` |
| 2 | Exact equivariance without augmentation. | Mean rotation error ~2e-6 (float32) over 32 rotations/question; float64 test < 1e-9 | `results.md`, `tests/` |
| 3 | Works on unseen wordings with a science text encoder. | Held-out wordings: force cos 0.917, dipole cos 0.941 (MatSciBERT) vs 0.859 / 0.726 (MiniLM) | `results.md` |
| 4 | Attention is mostly interpretable. | Peak on asked atom or a bonded neighbour 94-97%; exact-atom accuracy is seed-noisy (0.08-0.94) | `analysis/breakdown.md` |
| 5 | A reusable benchmark. | TensorQA-v0 (84,757 questions) with rotation protocol; compositional draft (84,196 questions, `scripts/13`) | `DATA_CARD.md` |
| 6 | Honest negatives. | Dipole: task ID slightly better (0.067 vs 0.082 MAE). Force: a rule-parser pipeline is as good or better than YLM (0.0057 vs 0.0064) | `results.md`, `results_baselines.md` |
| 7 | Invariant-only models cannot answer vector questions. | Scalar collapse: force cos 0.000 (3 seeds). True by construction; this only measures the gap | `results_baselines.md` |
| 8 (weak) | Pointer + CG-style combination helps on compositional questions. | T4 cos 0.89 vs 0.76, T5 0.78 vs 0.56 (Option B/C-lite vs A, 3 seeds, seen wordings) | `results_baselines.md` |

## Not supported (do not claim)

- That YLM beats a pipeline or an LLM. The pipeline matches it on v0; the Qwen baseline did not learn (so it says nothing).
- Generalisation of the compositional model to new wordings: held-out T4 cosine is 0.05 (Option A: -0.38).
- Anything about tensors (l=2) in the model (only quadrupole labels exist), crystals, chirality, or generation.
- Real-DFT accuracy: labels are GFN2-xTB.

## Closest related work

EquiLLM (invariants only into the LLM) and Mol-StrucTok, KDD '26 (invariant discrete tokens for generation).
Both drop l>=1 direction before the language side; see `02_novelty_check.md`.

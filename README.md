# YLM (Y_lm + Language Model)

## The idea in 3 sentences

1. We have a text encoder (reads papers) and a 3D encoder (reads atoms).
2. The 3D encoder keeps direction information (vectors, tensors), not just plain numbers.
3. We want text tokens to read those vectors/tensors directly, so the model can answer
   questions like "which way does the dipole point?" and the answer rotates when the
   molecule rotates.

## Files (read in this order)

| File | What is inside |
|---|---|
| `00_glossary.md` | Every math word used in the other files, explained simply. Read first if unsure. |
| `01_related_work.md` | List of papers, grouped by topic. One line each. |
| `02_novelty_check.md` | Has anyone done this already? Where do other models throw away direction info? |
| `03_technical_design.md` | How to build the Irreps-to-Text attention. 3 options. How to output a vector. Chirality. |
| `04_data_and_benchmark.md` | Datasets, a small benchmark, rotation test, baselines. |
| `05_risks.md` | Top 5 risks and what to do about each. |
| `06_plan_12_weeks.md` | Week-by-week plan. First-month experiment. |

## Honesty notes

- Paper list was written from memory plus a few web searches on 2026-09-29.
  arXiv IDs marked `(verify)` I am less sure about. Check them before citing.
- The novelty check is based on a small set of searches, not a full review.
  Re-run the searches in `02_novelty_check.md` before writing the paper.

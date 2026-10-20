# v1.0 plan (draft)

## Target venue
ICLR 2027 main track closed on 2026-09-25 (AoE) (https://www.iclr.cc/Conferences/2027/AuthorGuidelines).
Realistic targets: ICLR 2027 workshops (AI4Science-type, deadlines usually Jan-Feb), ICML 2027 main track (check
the date, usually late Jan), arXiv as soon as v1.0 is tagged. Everything below is venue-independent.

## What a reviewer will ask for, and the current gap
| Ask | State on 2026-10-19 | v1.0 work |
|---|---|---|
| Clear claim | "exact equivariance + wording generalisation"; the end-to-end win over a parser pipeline is NOT shown | Position as: pipeline needs a rule per wording (parser acc 100% -> 60% when wordings grow), YLM does not. Needs the cpipe_* runs |
| Real compositional generalisation | 2 held-out wordings, template text only | many held-out wordings, LLM-paraphrased free-form questions, 3+ seeds |
| Stronger LLM baseline | Qwen did not learn (not evidence) | fix (LR/format study), plus LLM->tool-call->parser baseline |
| Output beyond l=1 | l=2 labels not generated | quadrupole labels (PySCF), l=2 output head, rotation test with Wigner-D |
| Data scale/diversity | QM9 20k, xTB | add a second dataset (rMD17 or QM7-X), a DFT-label subset |
| Ablations | none | invariant-only attention, no gate, 1 vs 2 pointers, text encoder, data fraction 3/10/30% x 3 seeds |
| Honest limits | HIGHLIGHTS "Not supported" list | keep |

## Order of work (each on its own branch off main, merged when results are in)
1. Merge exp/pipeline-comp + exp/bc-heldout once cpipe_* finish (update results/HIGHLIGHTS first).
2. exp/l2-output: quadrupole labels -> T6 question -> l=2 head.
3. exp/free-form: paraphrased questions (needs an LLM paraphrase pass, check cost/consent), held-out paraphrase families.
4. exp/dataset2: second dataset.
5. exp/ablations: data-fraction sweep + ablation grid.
6. paper/: LaTeX draft, figures from scripts, reproduction table. Tag v1.0.0 and release checkpoints.

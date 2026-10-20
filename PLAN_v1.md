# v1.0 plan (draft)

## Target venue
ICLR 2027 main track closed on 2026-09-25 (AoE) (https://www.iclr.cc/Conferences/2027/AuthorGuidelines).
Realistic targets: ICLR 2027 workshops (AI4Science-type, deadlines usually Jan-Feb), ICML 2027 main track (check
the date, usually late Jan), arXiv as soon as v1.0 is tagged. Everything below is venue-independent.

## Revised priorities (after external review, 2026-10-19)
Weakest point: language may not be necessary. Our own v0 results: a parser pipeline beats YLM on force (0.0057 vs 0.0064),
a task-ID beats text on the dipole (0.067 vs 0.082). Language currently only picks the atom. Three additions, in order:
1. **Tasks only language can do** (relational and referring questions, e.g. "the hydrogen nearest the carbonyl oxygen",
   compositional T4/T5), with free-form LLM-paraphrased wordings as the MAIN metric, and a measured failure of the rule
   parser (already: 56-60% on free-form, 31% on held-out proj). Branch exp/free-form (templates generated, training pending).
2. **Parity / chirality.** Pseudoscalar (0o) targets, e.g. enantiomer sign, and axial-vs-polar vector outputs (1e vs 1o).
   Careful claim: a distance-only invariant model is blind to chirality; an SE(3)-equivariant model *can* output a
   pseudoscalar but nothing forces f(P x) = -f(x); an O(3) model with parity bookkeeping guarantees it exactly. Test the
   reflection error, not only the rotation error. Branch exp/parity.
3. **Real data and l=2.** SPICE or OMol25 (DFT): polarisability, quadrupole, NMR tensor questions; electrolyte-style
   questions (solvation shell around Li+). Quadrupole on QM9 (HF/6-31G*) is in progress on exp/l2-output as the first step.
Also: a small theoretical statement (exact equivariance of the invariant-query / equivariant-value readout, and why a
text-conditioned readout that is linear in the values is closed under O(3) with parity), and an anonymised mirror of the
repo for double-blind review (the repo is public and carries names).
Related work to cite: GeoMFormer (arXiv 2406.16853), GEM (arXiv 2406.15677); see 01_related_work.md.

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

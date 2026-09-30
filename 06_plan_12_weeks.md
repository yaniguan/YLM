# 12-Week Plan

## The one experiment that must work in month 1

**"Can a word select a direction?"**

- Data: 20k QM9 molecules. Compute dipole vectors and per-atom forces (or partial-charge vectors) with GFN2-xTB.
- Questions: T1 (dipole) + T3 ("force on the <element> atom <description>") from templates.
- Model: small e3nn encoder (or PaiNN, l <= 1) + small frozen text encoder (MiniLM or MatSciBERT) + Option A cross-attention + l=1 readout.
- Pass criteria:
  1. Equivariance test passes (error < 1e-4 in float32).
  2. On T3, clearly beats baseline v (one-hot task ID) and baseline ii (coordinates as text).
  3. Attention on T3 lands on the right atom most of the time.

If this fails, stop and rethink before building anything bigger.

## Week by week

| Week | Do | Done when |
|---|---|---|
| 1 | Read EquiLLM, EquiVLA, CatalyticMLLM, 3D-MoLM fully. Redo novelty searches. Set up e3nn, repo, equivariance test. | Novelty notes updated. Test runs on a random model. |
| 2 | Label pipeline: QM9 subset -> xTB -> dipole vector, forces, polarizability tensor. Question templates for T1, T3. | 20k molecules with labels. Q&A JSON file. |
| 3 | Build Option A. Train on T1 + T3. | Equivariance test passes. First numbers. |
| 4 | Baselines v (one-hot) and ii (coordinates as text, small LLM). **Month-1 checkpoint.** | Table: ours vs v vs ii on T1, T3, with rotation protocol. |
| 5 | Add Option B scores. Add T4 (projection) and T5 (comparison) questions. | A vs A+B ablation. |
| 6 | Add Option C (text-weighted tensor product) + l=2 readout. Polarizability tensor (T2). | Tensor results, equivariance test passes for l=2. |
| 7 | Baseline i-b (EquiLLM-style) and vi (frame averaging), same encoder. | Full baseline table on molecules. |
| 8 | Chirality tests (pseudoscalars on/off). Paraphrase + held-out template tests. | Chirality table. Generalization table. |
| 9 | Move to crystals: MACE-MP-0 encoder (frozen), MP dielectric tensors on Li/Na compounds, MPtrj forces. | Pipeline runs on crystals. |
| 10 | Battery text: COD/MP structures + abstracts, battery database matched to MP. Contrastive pretraining of text-structure alignment, then fine-tune on TensorQA. | Does pretraining help? One table. |
| 11 | Two-stream version with a small generative LLM (~0.5-1.5B, LoRA), `<VEC>`/`<TENSOR>` tokens. Baseline iii (3D-MoLM). | Generates text + vectors end to end. |
| 12 | Clean results, sample-efficiency curves, attention figures, write draft. | Paper draft + released benchmark v0. |

## Things to decide early (week 1)

- Encoder: e3nn custom (flexible, slower) vs MACE (strong, less flexible). Suggest: custom e3nn for molecules (weeks 3-8), MACE-MP-0 for crystals (weeks 9+).
- Text model: run MiniLM and MatSciBERT side by side from month 1 (decided). BatteryBERT added for crystals.
- float64 for tests, float32 for training.

## Target venue

ICML 2027 (deadline ~late Jan 2027, verify) if T3-T5 show a clear gap over EquiLLM-style and frame averaging.
If the gap is small, aim for a workshop (AI4Mat, ML4PS) + benchmark release first.

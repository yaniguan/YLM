# YLM: Letting Language Read 3D Directions

**YLM** (Y_lm + Language Model) lets text tokens attend directly to rotation-equivariant 3D features
(vectors and tensors), so a model can answer questions like *"what is the force on this oxygen atom?"*
with a vector that rotates correctly when the molecule rotates.

> **Status: v0.** First release: the **TensorQA-v0** benchmark (dipole and force questions on QM9),
> the Option A model, and three baselines. See [What is in v0](#what-is-in-v0) and [Results](#results-v0).

---

## The problem

Molecule/material language models today lose direction information before the language model sees it.

| Model | What the language model gets |
|---|---|
| MolT5, MoleculeSTM, MolCA | 1D SMILES or 2D graph. No 3D at all. |
| 3D-MoLM, Uni-Mol based | 3D, but built from pairwise distances, so invariant only. |
| CLaSP (crystal-text) | One pooled, invariant vector per structure. |
| EquiLLM (ICML 2025) | Equivariant encoder, but **only invariants go into the LLM**. Vectors bypass it. |
| LLMs on CIF / xyz text | Coordinates as text. No rotation guarantee. |

So language can't choose, combine, or report directions: dipoles, forces, dielectric or elastic tensors.
Details and evidence: [`02_novelty_check.md`](02_novelty_check.md).

## Core idea

```
question text ──► text encoder ──► h_t (invariant)   ── gives the QUERY
3D atoms      ──► SE(3) encoder ──► V_i (irreps)     ── gives KEYS and VALUES

                 Irreps-to-Text cross-attention
                              │
                              ▼
               e_t (irreps: l = 0, 1, 2, ...)  ── invariants of e_t are fed back into h_t
                              │
                              ▼
               equivariant readout ──► vector / tensor answer
```

One rule makes it work:

> **Attention weights use only invariant numbers. Anything with a direction is only multiplied by
> invariant numbers or combined through Clebsch-Gordan products.**

Then rotating the input by `R` rotates the output by `D(R)`, exactly, by construction.

Three building blocks, from simple to strong:

| Option | What text can do | Needed for |
|---|---|---|
| **A.** Invariant weights x equivariant values | pick atoms and channels | vectors (dipole, force on an atom) |
| **B.** Direction-aware keys (same-l dot products) | pick atoms by orientation | "component along this bond" |
| **C.** Text-weighted CG tensor product | build new tensors (1x1->2, ...) | polarizability, dielectric, elastic tensors |

Planned (not in v0): each text token carries two streams, the normal invariant hidden state `h_t` and an
irreps side stream `e_t`; to output a vector, a generative LLM emits a `<VEC>` / `<TENSOR>` token and an
equivariant head fills in the numbers.

Full design, pseudocode, chirality handling: [`03_technical_design.md`](03_technical_design.md).

## Main contributions (planned)

1. **Irreps-to-Text cross-attention.** The first layer (to our knowledge) where language tokens query
   l>=1 irreps directly and the output stays exactly SE(3)-equivariant.
2. **Two-stream token design.** Each token has an invariant language state and an equivariant geometry state,
   so a pretrained LLM can be used without breaking the symmetry.
3. **Chirality-aware alignment.** Keep parity labels and allow pseudoscalars into attention, so text like
   "(R)-" vs "(S)-" can be matched to the correct mirror image.
4. **TensorQA benchmark.** (Structure, question) -> vector/tensor, with questions where the text must
   select a direction, plus a rotation-consistency protocol (equivariance error under random rotations
   and random input frames).
5. **Controlled comparison** against scalar-collapse, EquiLLM-style adapters, coordinates-as-text LLMs,
   frame averaging, and a "text replaced by task ID" ablation.

The novelty claim is based on a limited search (2026-09-29). See the re-check list in
[`02_novelty_check.md`](02_novelty_check.md).

## What is in v0

| Part | In v0 | Later |
|---|---|---|
| Benchmark | **TensorQA-v0**: 84,757 questions on 20k QM9 molecules. Dipole of the molecule; force on an atom described in words. Held-out wordings. [Data card](DATA_CARD.md) | tensors (polarizability, dielectric), crystals, compositional questions |
| Model | equivariant encoder (l <= 2) + **Option A** Irreps-to-Text attention + l=1 readout. Frozen text encoder (MiniLM or MatSciBERT) | Options B and C, two-stream LLM, `<VEC>` tokens |
| Checkpoints | [`checkpoints/v0/`](checkpoints/v0/README.md): YLM + MatSciBERT, YLM + MiniLM, task-ID baseline (seed 0, 1-2 MB each) | larger models, more seeds |
| Baselines | task ID instead of text (done); coordinates as text into Qwen2.5-0.5B (queued, results pending); *post-v0.1.0, scripts ready, runs queued:* text→atom-index→force pipeline, invariants-only (scalar-collapse) | EquiLLM-style adapter, frame averaging, 3D-MoLM |
| Evaluation | rotation protocol: 32 random rotations per test question, equivariance error, attention-on-the-right-atom accuracy | mirror / chirality tests |

## Results (v0)

Full table with all runs: [`results.md`](results.md) (made by `scripts/07_make_table.py`).

Test set, mean ± std over seeds (5 seeds for YLM, 4 for task ID). Every test question is evaluated under
32 random rotations. MAE units: dipole e·bohr, force hartree/bohr. "Near" = the attention peak is on the asked
atom or an atom bonded to it; mean over seeds (range in brackets).

| Model | force MAE | force cos | dipole MAE | dipole cos | attention near asked atom | rotation error (mean) |
|---|---|---|---|---|---|---|
| **YLM + MatSciBERT** | **0.0064 ± 0.0003** | **0.960** | 0.082 | 0.980 | 0.97 (0.93–1.00) | 2e-6 |
| YLM + MiniLM | 0.0075 ± 0.0006 | 0.946 | 0.083 | 0.980 | 0.94 (0.87–0.98) | 2e-6 |
| Task ID instead of text (no language) | 0.0424 ± 0.0000 | 0.113 | **0.067** | **0.986** | 0.20 (0.12–0.43) | 2e-6 |
| Coordinates as text → Qwen2.5-0.5B | 0.0425 | ≈0 | 0.61 | ≈0 | – | – |
| Pipeline: rule parser → atom → equivariant force model (post-v0) | 0.0057 ± 0.0001 | 0.970 | 0.080 | 0.978 | 1.0 (given) | 3e-6 |

Held-out wordings (sentences never seen in training):

| Model | force cos | dipole cos |
|---|---|---|
| YLM + MatSciBERT | 0.917 ± 0.021 | 0.941 ± 0.032 |
| YLM + MiniLM | 0.859 ± 0.040 | 0.726 ± 0.140 |

Post-v0 baselines ([`results_baselines.md`](results_baselines.md)) change how to read this table:
the Qwen baseline did **not learn the force at all** in any setting tried (it can memorise 16 questions, so the code works,
but it does not generalise; tuning was narrow), so it is **not** evidence that an end-to-end LLM cannot do this.
And a hand-written rule parser feeding the same equivariant encoder (100% parse accuracy on v0 questions) is
**as good or better than YLM** (force MAE 0.0057 vs 0.0064). v0 questions are too easy to parse to separate them.

What this shows:

1. **The words select the atom.** Without the question text (task ID only), the model cannot know which atom
   is asked about, and the force direction is close to random (cos 0.11). With the text, force error drops
   6.6x and the direction is right (cos 0.96).
2. **The answer rotates exactly with the molecule.** Mean rotation error is ~2e-6 (float32 rounding) for every
   model, without any rotation augmentation. Worst cases reach ~1e-3 on near-zero predictions, where the
   relative error is large.
3. **A science text encoder generalizes better.** MatSciBERT beats MiniLM on held-out wordings, most clearly on
   the dipole (0.94 vs 0.73). With very little data (3-10%) MiniLM learns faster; from 30% on MatSciBERT is better
   ([plot](analysis/sample_efficiency.png)).
4. **Attention is local, not always exact.** For hydrogen atoms the attention almost never peaks on the H itself
   but on the heavy atom it is bonded to (98% "near", 0-40% exact). Some seeds do the same for oxygen. The
   neighbour's features already carry the force on the asked atom after message passing, so the answer is still
   right. Details: [`analysis/breakdown.md`](analysis/breakdown.md).
5. **Where language costs something.** For the dipole, a one-hot task ID is slightly better (MAE 0.067 vs 0.082):
   the dipole question needs no atom selection, and the text model must first recognize the question type.

Full tables, including 3% / 10% / 30% training data: [`results.md`](results.md).

## Limitations

- Labels are GFN2-xTB (semi-empirical), not DFT. Fine for testing whether a model reads geometry; not chemical accuracy.
- Questions come from 12 templates (4 held out). Real questions are more varied.
- Only vectors (l = 1) are asked in v0, although the encoder carries l = 2.
- The force question can also be solved by a pipeline (**confirmed post-v0: it matches or beats YLM on v0 questions**, see above)
  -- original note: "parse the description -> atom index -> force model".
  v0 shows the language model can do it end to end; harder compositional questions are needed to rule out simpler pipelines.
- The text encoder is frozen and only reads the question; the model does not generate text yet.
- Attention often lands on a bonded neighbour instead of the exact atom (see Results), so attention maps are
  only roughly interpretable.
- Data-fraction runs (3%, 10%, 30%) use one seed each.

## First experiment

**Can a word select a direction?**

- 20k QM9 molecules, dipole vectors and per-atom forces recomputed with GFN2-xTB.
- Questions like *"What is the force on the oxygen atom bonded to two carbons?"*
- Small e3nn encoder + small frozen text encoder + Option A + l=1 readout.
- Pass if: equivariance error < 1e-4, clear win over the task-ID and coordinates-as-text baselines,
  and attention lands on the right atom.

## Repository layout

```
YLM/
├── README.md                 # this file
├── LICENSE                   # MIT (code)
├── 00_glossary.md            # irreps, CG products, parity: plain explanations
├── 01_related_work.md        # papers by topic, one line each
├── 02_novelty_check.md       # closest work, where others lose 3D direction
├── 03_technical_design.md    # attention options A/B/C, output heads, chirality
├── 04_data_and_benchmark.md  # datasets, TensorQA, rotation protocol, baselines
├── 05_risks.md               # top 5 risks and fixes
├── 06_plan_12_weeks.md       # week-by-week plan
├── progress.md               # dated progress log
├── DATA_CARD.md              # TensorQA-v0: sources, units, splits, limits
├── results.md                # v0 results table
├── environment.yml           # conda environment
├── src/                      # encoder, Irreps-to-Text attention, model
├── scripts/                  # 01 labels, 02 questions, 03 text features, 04 train, 05 LLM baseline,
│                             # 06 export release data, 07 results table, 08 analysis plots,
│                             # 09 evaluate a checkpoint + rotation demo, 10 package checkpoints
├── analysis/                 # sample-efficiency plot, error breakdown
├── checkpoints/v0/           # 3 small trained checkpoints (MatSciBERT, MiniLM, task-ID) + README
├── tests/                    # equivariance tests
├── jobs/                     # Hoffman2 (SGE) and Expanse (Slurm) job scripts
└── data/                     # downloaded / generated data (git-ignored)
```

## How to run

```bash
# 1. environment (Mac or Linux; for old Linux see the note in environment.yml)
conda env create -f environment.yml
conda activate ylm

# 2. data: QM9 structures (DeepChem mirror), ~45 MB
mkdir -p data && cd data
curl -LO https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip && unzip qm9.zip && cd ..

# 3. labels, questions, text features
python scripts/01_make_labels.py --n 20000      # GFN2-xTB dipoles + forces (~15 s)
python scripts/02_make_questions.py             # TensorQA-v0 questions
python scripts/03_embed_text.py --name minilm
python scripts/03_embed_text.py --name matscibert

# 4. check equivariance, then train
python -m pytest tests -q
python scripts/04_train.py --text minilm --name ylm_minilm      # ours
python scripts/04_train.py --text taskid --name base_taskid     # baseline: no language
python scripts/05_train_llm_baseline.py --name llm05 --augment  # baseline: coordinates as text (GPU)

# 5. (optional) skip training: evaluate the released v0 checkpoint and see the rotation demo
python scripts/09_eval_checkpoint.py --ckpt checkpoints/v0/ylm_matscibert_s0.pt --demo

# 6. table and release files
python scripts/07_make_table.py           # -> results.md
python scripts/08_analyze.py              # -> analysis/ (plot + breakdown by element, size, wording)
python scripts/06_export_release.py       # -> data/release/tensorqa_v0.tar.gz (JSONL + data card)
```

Only want the benchmark? Download `tensorqa_v0.tar.gz` from the GitHub release. Each JSONL line has the atoms,
positions, question and answer, so no code from this repo is needed.

Results go to `runs/<name>/results.json`. Cluster job scripts are in `jobs/`.
Progress notes: [`progress.md`](progress.md).

## Roadmap

- [x] v0: labels, TensorQA-v0, Option A, rotation protocol, task-ID and coordinates-as-text baselines
- [x] v0.1.0 release: checkpoints, `scripts/09` checkpoint eval + rotation demo (GitHub release v0.1.0)
- [~] After v0.1.0 (code written and smoke-tested, **no results yet**; jobs queued on Hoffman2):
  Qwen coordinates-as-text baseline (`jobs/hoffman2_llm.sh`), pipeline baseline (`--model pipeline`, rule parser
  `src/parse_question.py`, 100% parse accuracy on v0 questions, so v0 cannot separate it from YLM),
  invariants-only baseline (`--model scalar`), compositional questions T4 diff / T5 proj
  (`scripts/13`, 84,196 questions) with Option B/C-lite (`src/option_bc.py`, `scripts/14`),
  l=2 quadrupole labels (`scripts/12`, PySCF; rotation check 6e-12), extra seeds for the data-fraction curve (`jobs/sweep_fractions.sh`, not submitted)
- [ ] Weeks 5-8: tensor outputs (l=2) in the model, EquiLLM-style and frame-averaging baselines, chirality tests
- [ ] Weeks 9-10: crystals (MACE-MP-0, Materials Project tensors), battery literature text
- [ ] Weeks 11-12: generative LLM version, paper draft

## Development workflow

`main` is the released state (tag `v0.1.0`). New work goes on a branch from `main` (`feat/<name>`, `exp/<name>`),
is merged back when its results are in `results.md`, and each release gets a tag and a GitHub release.

## Built on

[e3nn](https://github.com/e3nn/e3nn) ·
[MACE](https://github.com/ACEsuit/mace) ·
[Materials Project](https://materialsproject.org) ·
[QM9](https://doi.org/10.1038/sdata.2014.22) ·
[xTB](https://github.com/grimme-lab/xtb)

## Citation

Not published yet. Placeholder:

```bibtex
@misc{ylm2026,
  title  = {YLM: Irreps-to-Text Cross-Attention for Equivariant Language-Geometry Alignment},
  author = {Guan, Yani},
  year   = {2026},
  note   = {Work in progress}
}
```

## License

- Code: [MIT](LICENSE)
- Data and benchmark (TensorQA): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

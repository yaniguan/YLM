# YLM: Letting Language Read 3D Directions

**YLM** (Y_lm + Language Model) lets text tokens attend directly to rotation-equivariant 3D features
(vectors and tensors), so a model can answer questions like *"what is the force on this oxygen atom?"*
with a vector that rotates correctly when the molecule rotates.

> **Status:** month 1 of the [12-week plan](06_plan_12_weeks.md). Data pipeline, model (Option A) and
> equivariance tests work; first training runs in progress. See [`progress.md`](progress.md).

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

Each text token carries two streams: the normal invariant hidden state `h_t`, and an irreps side stream `e_t`.
To output a vector, the model emits a `<VEC>` / `<TENSOR>` token and an equivariant head fills in the numbers.

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
├── src/                      # encoder, Irreps-to-Text attention, model
├── scripts/                  # 01 labels, 02 questions, 03 text features, 04 train, 05 LLM baseline
├── tests/                    # equivariance tests
├── jobs/                     # Hoffman2 (SGE) and Expanse (Slurm) job scripts
└── data/                     # downloaded / generated data (git-ignored)
```

## How to run

```bash
# 1. environment (Mac or Linux)
conda create -n ylm -c conda-forge python=3.11 pytorch tblite-python rdkit ase -y
conda activate ylm
pip install e3nn transformers pytest

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
```

Results go to `runs/<name>/results.json`. Cluster job scripts are in `jobs/`.
Progress notes: [`progress.md`](progress.md).

## Roadmap

- [ ] Weeks 1-4: labels, Option A, rotation test, first baselines (month-1 checkpoint)
- [ ] Weeks 5-8: Options B and C, tensor outputs, EquiLLM-style and frame-averaging baselines, chirality tests
- [ ] Weeks 9-10: crystals (MACE-MP-0, Materials Project tensors), battery literature text
- [ ] Weeks 11-12: generative LLM version, paper draft, TensorQA v0 release

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

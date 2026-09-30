# YLM: Letting Language Read 3D Directions

**YLM** (Y_lm + Language Model) lets text tokens attend directly to rotation-equivariant 3D features
(vectors and tensors), so a model can answer questions like *"what is the force on this oxygen atom?"*
with a vector that rotates correctly when the molecule rotates.

> **Status:** research planning stage (week 0). No code yet. This repo holds the design notes and
> literature review. Code will be added in `src/` following the [12-week plan](YLM/06_plan_12_weeks.md).

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
Details and evidence: [`YLM/02_novelty_check.md`](YLM/02_novelty_check.md).

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

Full design, pseudocode, chirality handling: [`YLM/03_technical_design.md`](YLM/03_technical_design.md).

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
[`YLM/02_novelty_check.md`](YLM/02_novelty_check.md).

## First experiment

**Can a word select a direction?**

- 20k QM9 molecules, dipole vectors and per-atom forces recomputed with GFN2-xTB.
- Questions like *"What is the force on the oxygen atom bonded to two carbons?"*
- Small e3nn encoder + small frozen text encoder + Option A + l=1 readout.
- Pass if: equivariance error < 1e-4, clear win over the task-ID and coordinates-as-text baselines,
  and attention lands on the right atom.

## Repository layout

```
.
├── README.md                     # this file
└── YLM/
    ├── README.md                 # reading order for the notes
    ├── 00_glossary.md            # irreps, CG products, parity: plain explanations
    ├── 01_related_work.md        # papers by topic, one line each
    ├── 02_novelty_check.md       # closest work, where others lose 3D direction
    ├── 03_technical_design.md    # attention options A/B/C, output heads, chirality
    ├── 04_data_and_benchmark.md  # datasets, TensorQA, rotation protocol, baselines
    ├── 05_risks.md               # top 5 risks and fixes
    └── 06_plan_12_weeks.md       # week-by-week plan
```

## How to run

There is no code yet. Planned environment for the first experiment:

```bash
conda create -n ylm python=3.11 -y
conda activate ylm
pip install torch e3nn transformers ase
conda install -c conda-forge xtb-python   # for dipole / force labels
```

To read the notes, start with [`YLM/README.md`](YLM/README.md).

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

Not chosen yet. Suggested: MIT for code, CC BY 4.0 for the benchmark (matches Materials Project data).

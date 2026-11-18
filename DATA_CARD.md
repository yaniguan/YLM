# Data card: TensorQA-v0

## What it is

Questions about a 3D molecule whose answer is a **3D vector**.
Input: atoms (element + position) and an English question. Output: 3 numbers.

| Task | Example question | Answer |
|---|---|---|
| `dipole` | "What is the dipole moment of this molecule?" | dipole vector of the whole molecule |
| `force` | "What is the force on the oxygen atom bonded to two carbons?" | force vector on that one atom |

For `force`, the atom is described **only in words** (element + bonded neighbours, plus "in a ring" if needed).
Each description matches exactly one atom in the molecule. `atom_index` is given for evaluation only;
a model should not use it as input.

## Size

| Split | Questions | Molecules | Notes |
|---|---|---|---|
| train | 61,619 | ~15.9k | train templates |
| val | 7,732 | ~2.0k | train templates |
| test | 7,703 | ~2.0k | train templates |
| test_heldout | 7,703 | same molecules as test | **new wordings never seen in training** |

Molecules are split at random (80/10/10). No molecule appears in two of train/val/test.

## Official v0.1.0 hashes and counts

The packaged v0 benchmark on the [`v0.1.0` release](https://github.com/yaniguan/YLM/releases/tag/v0.1.0) has **84,757 questions**, **584 unique question strings**, and **19,843 molecules**:

| Artifact | Questions | SHA-256 |
|---|---|---|
| `tensorqa_v0.tar.gz` (release archive) | 84,757 | `8a51673f849c217febeebcbb934c40c2fab6dcc28884e689b99b6cd36c85c782` |
| `tensorqa_v0/train.jsonl` | 61,619 | `1ac8fbd9da6135b55a6aba43a89a2c5e0cbb1d0765d92f492a6996288902c68f` |
| `tensorqa_v0/val.jsonl` | 7,732 | `0b0cb4baae8d31b2b0ec5a0d35931ed68414236b7b7646445bbcafff001ce7fd` |
| `tensorqa_v0/test.jsonl` | 7,703 | `24bf8adffdee1af126f914dce912c0c5d3a1a5143eea6f2bbdfe4a470791139b` |
| `tensorqa_v0/test_heldout.jsonl` | 7,703 | `5f223d8f46917d63f2eb3abb1280c82914a29990ea27042bc09de57bebcbeffe` |
| `data/tensorqa_v0.json` (unpacked via `scripts/06b_unpack_release.py`) | 84,757 | `76cf64ea621b5c5e521caeb1bbc6511bfb4b0fac7a521e79d67f5249a73630d3` |
| Canonical `(split, task, atom, question)` stream | 84,757 | `e771153eb1de961de74dc82b7a4cae2001280fb7e8aa60bf05a62a430dcb6c34` |

Check a local copy with `python -m src.provenance --check-v0 data/tensorqa_v0.json` (or pass `tensorqa_v0.tar.gz`).
To unpack the official v0 training files without running `tblite`, run `python scripts/06b_unpack_release.py --download`.

**Important:** Rebuilding from scratch with `scripts/01_make_labels.py` and `scripts/02_make_questions.py` produces a **different** question file across `tblite` / `RDKit` builds (for example, the rebuild in [`docs/WORKLOG_oct29b.md`](docs/WORKLOG_oct29b.md) gave `84,787` questions and `571` unique strings). Runs trained or evaluated on a fresh rebuild must **not** be compared directly to the official v0 results table.

## Where the numbers come from

1. **Structures**: 20,000 random molecules from QM9 (Ramakrishnan et al., 2014), via the DeepChem mirror
   (`gdb9.sdf`). Up to 29 atoms; elements H, C, N, O, F.
2. **Displacement**: every atom moved by Gaussian noise (sigma = 0.05 Angstrom, fixed seed). QM9 geometries are
   relaxed, so forces there are ~0; the noise makes forces meaningful.
3. **Labels**: GFN2-xTB single point (tblite 0.7). Dipole in e*bohr, forces in hartree/bohr (= -gradient).
4. Dropped: 1 molecule where xTB failed, 156 molecules where any force component was above 0.5 hartree/bohr
   (the noise put two atoms too close).

Original generation scripts: `scripts/01_make_labels.py`, `scripts/02_make_questions.py` (seed 0). To unpack the exact official v0 files instead of running a fresh rebuild, use `python scripts/06b_unpack_release.py --download`.

## Units and frames

- Positions: bohr. Dipole: e*bohr. Force: hartree/bohr.
- All molecules are neutral, so the dipole does not depend on the origin.
- Vectors are given in the same frame as the positions. If you rotate the positions by R, rotate the answer by R.

## How to evaluate (rotation protocol)

For each test question, run the model on the molecule under K = 32 random rotations (the first one is no
rotation). Report, per task:
- `mae`: mean absolute error per vector component (physical units)
- `cosine`: cosine between predicted and true vector
- `eq_err`: ||f(Rx) - R f(x)|| / ||f(x)|| (0 for an exactly equivariant model)
- `atom_acc` (force only, attention models): does the attention peak on the asked atom?

Also report the error in the dataset's own orientation (`mae_orig_frame`) for models that are not equivariant.

## Known limitations

- **Labels are semi-empirical (GFN2-xTB), not DFT.** Good enough to test whether a model can *read* geometry,
  not to claim chemical accuracy.
- **Questions are templated** (6 dipole and 6 force templates, 2 of each held out). Real users ask more varied things.
- **Only vectors (l = 1).** Tensors (polarizability, dielectric) come in a later version. xTB polarizabilities are
  wrong (water: 2.4 vs 9.8 bohr^3), so tensor labels need DFT or an existing dataset (e.g. Wilkins et al. 2019 tensors on QM7b molecules; the plain QM7b file only has scalar polarizabilities).
- **Small molecules only** (QM9). No crystals yet.
- The `force` question is solvable by "parse the description -> pick the atom -> predict its force".
  v0 tests whether a model can do this end to end from language; harder compositional questions come later.

## License

CC BY 4.0. QM9 is released under CC0 / open terms by its authors; please also cite:
Ramakrishnan, Dral, Rupp, von Lilienfeld, *Quantum chemistry structures and properties of 134 kilo molecules*,
Scientific Data 1, 140022 (2014). xTB: Bannwarth, Ehlert, Grimme, JCTC 15, 1652 (2019).

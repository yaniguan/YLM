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


# TensorQA-v1 and TensorQA-pol

## TensorQA-v1 (`data/tensorqa_v1.json`, built by `scripts/54_make_splits.py` and `scripts/55_make_tensorqa_v1.py`)

- **Molecules**: the same 20,000 QM9 molecules and GFN2-xTB forces as v0 (interface benchmark, not DFT accuracy).
- **Tasks** (3-vectors, hartree/bohr): `force` (`F_A`); `diff` (`F_A - F_B`); `proj` (`(F_A . u) u` with `u = unit(r_B - r_A)`, the *signed axis projection*, pointing away from `B` when `F_A . u < 0`).
- **Splits** (leak-aware):
  - Molecules with the same canonical heavy-atom graph never cross split boundaries (v0 shared 4 canonical graphs).
  - `scaf` holds out entire Bemis-Murcko scaffold groups (in v0, 86.1% of test molecules had a scaffold seen in train).
  - `test_heldout` and `scaf_heldout` re-word the test questions in two held-out style families (`chatty`, `inverted`).
  - `comb` holds out the `(task, element of A)` pairs `(proj, O)` and `(diff, N)` from train/val/test.
  - `cf` contains counterfactual groups (antisymmetry `D(A,B) = -D(B,A)` and chain additivity `D(A,C) = D(A,B) + D(B,C)`) on test molecules.
- **Wordings & Audit**: generated in 6 style families (`scripts/53_paraphrase_templates.py`), checked by a second pass, and audited (`scripts/57_audit_templates.py`) keeping only `ok` templates without false bond assumptions. `audit/audit_sample_200.csv` provides a 198-row stratified sheet for human review (`verdict` column left blank).
- **Rule parser baseline**: implemented in `src/parse_question.py` (`scripts/50_parse_comp.py`) by inspecting failures on the `train` split only; accuracy on held-out style families is reported alongside model metrics (`scripts/56_train_v1.py`, `scripts/61_analyze_v1.py`).

## TensorQA-pol (`data/tensorqa_pol.json`, built by `scripts/58_make_pol_data.py`)

- **Source & License**: Yang, Lao, Wilkins, Grisafi, Ceriotti, DiStasio, *Quantum Mechanical Static Dipole Polarizabilities in the QM7b and AlphaML Showcase Databases*, Scientific Data 6, 152 (2019), and Wilkins et al., *Accurate molecular polarizabilities with coupled cluster theory and machine learning*, PNAS 116, 3401 (2019); Materials Cloud Archive `10.24435/materialscloud:2019.0002`, **CC BY 4.0**. Contains 7,211 QM7b molecules (`C, H, N, O, S, Cl`) with LR-CCSD / d-aug-cc-pVDZ static polarizability tensors (plus B3LYP and SCAN0 tensors used only as classical quantum-chemistry references). Raw `qm7b_coords.xyz` is downloaded directly from Materials Cloud (`scripts/58_make_pol_data.py` header) and is not redistributed in git.
- **Tasks**: `pol_iso` (`tr(alpha)/3`, `0e` scalar), `pol_aniso` (traceless part, five `2e` coefficients), `pol_dir` (`u^T alpha u`, scalar), and `pol_ind` (`alpha u`, `1o` vector), where `u = unit(r_B - r_A)` between two word-described atoms `A` and `B`. Units: atomic units.
- **Splits**: grouped 80/10/10 by canonical connectivity graph (distance-based RDKit bonds used only to word atom descriptions; models receive atomic numbers and 3D positions only), plus `test_heldout` in held-out style families (`scripts/59_train_pol.py`, `scripts/60_parse_pol.py`).

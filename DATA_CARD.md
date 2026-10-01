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
584 unique question strings.

## Where the numbers come from

1. **Structures**: 20,000 random molecules from QM9 (Ramakrishnan et al., 2014), via the DeepChem mirror
   (`gdb9.sdf`). Up to 29 atoms; elements H, C, N, O, F.
2. **Displacement**: every atom moved by Gaussian noise (sigma = 0.05 Angstrom, fixed seed). QM9 geometries are
   relaxed, so forces there are ~0; the noise makes forces meaningful.
3. **Labels**: GFN2-xTB single point (tblite 0.7). Dipole in e*bohr, forces in hartree/bohr (= -gradient).
4. Dropped: 1 molecule where xTB failed, 156 molecules where any force component was above 0.5 hartree/bohr
   (the noise put two atoms too close).

Reproduce: `scripts/01_make_labels.py`, `scripts/02_make_questions.py` (seed 0).

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
  wrong (water: 2.4 vs 9.8 bohr^3), so tensor labels need DFT or an existing dataset (e.g. QM7b).
- **Small molecules only** (QM9). No crystals yet.
- The `force` question is solvable by "parse the description -> pick the atom -> predict its force".
  v0 tests whether a model can do this end to end from language; harder compositional questions come later.

## License

CC BY 4.0. QM9 is released under CC0 / open terms by its authors; please also cite:
Ramakrishnan, Dral, Rupp, von Lilienfeld, *Quantum chemistry structures and properties of 134 kilo molecules*,
Scientific Data 1, 140022 (2014). xTB: Bannwarth, Ehlert, Grimme, JCTC 15, 1652 (2019).

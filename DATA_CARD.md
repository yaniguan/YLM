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
  wrong (water: 2.4 vs 9.8 bohr^3), so tensor labels need DFT or an existing dataset (e.g. Wilkins et al. 2019 tensors on QM7b molecules; the plain QM7b file only has scalar polarizabilities).
- **Small molecules only** (QM9). No crystals yet.
- The `force` question is solvable by "parse the description -> pick the atom -> predict its force".
  v0 tests whether a model can do this end to end from language; harder compositional questions come later.

## License

CC BY 4.0. QM9 is released under CC0 / open terms by its authors; please also cite:
Ramakrishnan, Dral, Rupp, von Lilienfeld, *Quantum chemistry structures and properties of 134 kilo molecules*,
Scientific Data 1, 140022 (2014). xTB: Bannwarth, Ehlert, Grimme, JCTC 15, 1652 (2019).


# TensorQA-v1 and TensorQA-pol (added 2026-10-19)

## TensorQA-v1 (`data/tensorqa_v1.json`, built by `scripts/20_make_splits.py` and `scripts/21_make_tensorqa_v1.py`)
- **Molecules**: the same 20,000 QM9 molecules and GFN2-xTB forces as v0 (not DFT; 0.0064 Eh/bohr is about 0.33 eV/A,
  an interface-level number, not electronic-structure accuracy).
- **Tasks** (3-vectors, hartree/bohr): `force` F_A; `diff` F_A - F_B; `proj` (F_A.u)u with u = unit(r_B - r_A), the
  *signed axis projection* (it points away from B when F_A.u < 0; it is not a ray component).
- **Splits** (leak-aware): molecules with the same canonical heavy-atom graph never cross splits (v0 had 4 shared graphs);
  `scaf` holds out whole Bemis-Murcko scaffold groups (v0: 86.1% of test molecules had a scaffold seen in train);
  `test_heldout` / `scaf_heldout` re-word the same questions in two held-out style families; `comb` holds out the
  (task, element of A) pairs (proj, O) and (diff, N) from train/val/test; `cf` are counterfactual groups
  (antisymmetry, additivity) on test molecules. Rows record the wording `style` family.
- **Wordings**: written by `claude-haiku-5-5` in 6 style families (`scripts/19_paraphrase_templates.py`), checked by a
  second call of the same model, then audited by `claude-sonnet-5-5` (`scripts/23_audit_templates.py`); only templates the
  audit called `ok` (and with no "bond" wording, since A and B need not be bonded) are kept: 66 of the generated
  `force/diff/proj/quad` templates were removed (`data/paraphrases_removed.json`, raw file `data/paraphrases_raw.json`).
  This is not a human audit. `audit/audit_sample_200.csv` is the sheet for one (200 filled-in questions, stratified by task and style; the `verdict` column is empty).
- **Known ambiguities**: "difference between X and Y" is read as X - Y (the audit flagged such templates as ambiguous and
  they were removed from the free-form set, but the hand-written v0/comp templates still use it); some paraphrases can be
  read as a scalar component. Treat the last digits of wording-level results accordingly.
- **Rule parser baseline**: written by reading failures on the *train* split only, in two rounds (`src/parse_question.py`).
  Its accuracy on held-out style families is reported next to every result that uses it.

## TensorQA-pol (`data/tensorqa_pol.json`, `scripts/24_make_pol_data.py`)
- **Source**: Yang et al., *Quantum Mechanical Static Dipole Polarizabilities in the QM7b and AlphaML Showcase Databases*,
  Sci. Data 6, 152 (2019); Materials Cloud 10.24435/materialscloud:2019.0002, CC BY 4.0. 7,211 QM7b molecules (C, H, N, O, S, Cl),
  LR-CCSD / d-aug-cc-pVDZ polarizability tensors (also B3LYP and SCAN0 in the file, used only as a classical reference).
  The raw file is not redistributed here: download it with the command in the script header.
- **Tasks**: `pol_iso` (alpha_iso, scalar), `pol_aniso` (traceless part, five 2e coefficients), `pol_dir` (u^T alpha u),
  `pol_ind` (alpha u), where u points from atom A to atom B (two atoms described in words). Units: atomic units.
- **Splits**: by canonical connectivity graph (distance-based bonds from RDKit, used only to word the atom descriptions);
  no scaffold split (bond perception on bare coordinates is not reliable enough); `test_heldout` as above.

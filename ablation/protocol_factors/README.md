# Protocol-factor ablation

This study asks which inputs change machine reasoning on lab protocols:
protocol QA, step error detection, and step ordering or planning.

Three questions:

1. Do extra information layers help? The layers are first-principles descriptors
   (Q: GFN2-xTB), molecular foundation-model outputs (F: MoLFormer-XL heads and
   RDKit descriptors), solvation and electrolyte facts (S), and microenvironment
   facts (M: stated pH and temperature, tabulated pKa, melting and boiling
   points, ionic strength).
2. Does a formal form help? The forms are raw text, a structured scientific
   form (R2), LabLean code (R3), and LabLean code plus the checker verdict (R4).
3. Are native inputs (numbers or embeddings fed directly) as good as or better
   than the same facts written as text?

Only measured numbers are reported. Cells that were not run are marked **not run**.

## Factor design

**Information layers (factorial).** Every subset of {Q, F, S, M}, including
none (16 cells). All of these cells keep molecule identity (name and SMILES)
except the extra control A (no identity). Two further controls use the same
token length as all four layers: ALL-shuf (facts taken from a different
molecule) and ALL-plac (length-matched record metadata that is true but not
the layer facts).

Facts are computed from SMILES and stated conditions only. They never see the
question or the answer. Layer F heads were trained on public MoleculeNet ESOL
and Lipophilicity; 5-fold CV on those training sets is RMSE 0.86, R2 0.83 for
logS (n = 1128) and RMSE 0.86, R2 0.49 for logD (n = 4200). Those CV numbers
are not protocol-task results.

**Representation.** R1 is the original text. R2 is a structured form
(containers, step balances, reactions, state). R3 is LabLean code for items
that are liquid-handling expressible; others are marked not expressible. R4 is
R3 plus the Lean checker verdict. The converter is told to transcribe
faithfully and not to correct or flag errors.

**Modality.** Text: the same facts written in words. Native: the same facts as
standardized numbers (and, for CALiSol, salt/solvent identity). Both:
concatenation.

## Benchmarks

| Benchmark | License | What is in this repo | Status |
|---|---|---|---|
| ChemProto-Twins (own set) | written for this study | 84 scenarios, 336 items (`data/items_own.jsonl`, source `data/scenarios_own.json`) | items ready; reader runs **not run** |
| CALiSol-23 conductivity pairs | CC-BY 4.0, derived | 3000 / 251 / 300 train/dev/test pairs | 16-cell layer × modality probe measured; soft-token arms **not run** |
| LAB-Bench ProtocolQA public | CC-BY-SA 4.0 | scripts only; no item text | **not run** |
| BioProBench ERR subset | CC-BY-NC 4.0, research only | scripts only; ids and names only, no item text | **not run** |

ChemProto-Twins has six labels (qm, fm, solvation, micro, formal, none) times
14 scenarios. Each scenario has a correct protocol, a seeded-error twin, a
decoy (a harmless edit), and one multiple-choice item (protocol QA or
planning). Splits are by scenario, so twins stay together: 72 scenarios / 284
items in test, 12 scenarios / 52 items in dev. The set is synthetic. 68 of
the 336 items are marked LabLean-expressible.

CALiSol pairs ask which of two liquid-electrolyte formulations has the higher
ionic conductivity. Pairs come from the same source paper, have temperatures
within 1 K, and differ by at least 10% relative conductivity. The split is by
salt-solvent system, so test systems are held out.

## Readers and statistics

- Linear logistic and kNN (k = 15) probes on CALiSol, five seeds (0, 1, 2, 3,
  4). Each seed trains on a random half of the training pairs. Text facts are
  encoded with a frozen all-MiniLM-L6-v2 sentence embedding. Native facts are
  a standardized number vector plus salt/solvent identity. A/B order is
  flipped at training time so the probe cannot use answer position.
- Qwen2.5-0.5B-Instruct scored by answer log-likelihood, a frozen-LM
  soft-token arm (N0–N3), and a larger CLI model on a subset of cells:
  **not run** in this report.
- Accuracy uses a cluster bootstrap 95% CI (2000 resamples) over items
  (scenario groups on the own set; items on CALiSol). Paired differences use
  a cluster sign-flip p-value (5000 flips). Holm adjustment is applied to the
  list of terms in the effect table. Main effects average over matched cells
  in the full 16-cell layer factorial.

GPU numbers are not reported. All measured figures below are CPU runs.

## Measured results: CALiSol-23 probes

N = 300 held-out test pairs. Seeds = 0, 1, 2, 3, 4. All 16 layer subsets have
been run, for three modalities and two probes (96 cells).

Chance is 50%. The none / native / logistic cell is 63.5% [57.5, 68.6], so
identity, temperature, and concentration already beat chance before any of
Q, F, S, or M is added.

### Accuracy by cell

Values are the mean over five seeds, with a cluster bootstrap 95% CI on the
item-level seed-averaged scores. Seed SD is the standard deviation of the
five seed means, in percentage points, for the native arm.

| Layers | Probe | Native acc. % [95% CI] | Text acc. % [95% CI] | Both acc. % [95% CI] | Seed SD (native, pp) |
|---|---|---|---|---|---|
| none | logreg | 63.5 [57.5, 68.6] | 63.1 [58.5, 67.8] | 64.1 [59.9, 68.4] | 0.96 |
| none | knn | 68.3 [63.4, 73.2] | 63.6 [58.8, 68.1] | 64.4 [59.5, 69.1] | 1.35 |
| Q | logreg | 64.3 [58.7, 69.3] | 68.0 [63.4, 72.6] | 64.4 [59.9, 69.0] | 0.57 |
| Q | knn | 70.3 [65.5, 74.9] | 63.7 [59.1, 68.4] | 65.9 [61.4, 70.5] | 1.29 |
| F | logreg | 72.7 [67.3, 77.7] | 65.3 [60.5, 69.7] | 67.3 [62.6, 71.7] | 0.60 |
| F | knn | 73.2 [68.5, 77.6] | 65.7 [60.9, 70.6] | 66.4 [61.6, 70.9] | 1.36 |
| S | logreg | 62.7 [57.0, 67.9] | 67.5 [62.9, 72.2] | 66.1 [61.9, 70.5] | 0.93 |
| S | knn | 69.2 [64.5, 73.9] | 64.7 [60.1, 69.0] | 65.7 [61.2, 70.3] | 1.34 |
| M | logreg | 63.7 [58.0, 68.5] | 57.9 [53.1, 62.9] | 57.9 [53.7, 62.6] | 0.30 |
| M | knn | 67.7 [62.9, 72.5] | 63.7 [58.9, 68.5] | 64.6 [60.1, 69.1] | 0.88 |
| QF | logreg | 72.2 [66.9, 77.3] | 67.5 [62.9, 72.2] | 66.7 [62.3, 71.0] | 0.69 |
| QF | knn | 72.2 [67.6, 76.7] | 62.1 [57.3, 66.6] | 65.5 [61.3, 70.1] | 0.88 |
| QS | logreg | 62.8 [57.2, 67.7] | 64.9 [60.3, 69.8] | 62.9 [58.5, 67.1] | 1.09 |
| QS | knn | 69.7 [64.9, 74.4] | 63.4 [58.7, 68.0] | 65.3 [60.7, 69.9] | 1.68 |
| QM | logreg | 64.3 [58.7, 69.1] | 64.4 [59.8, 69.0] | 61.7 [57.5, 66.1] | 0.71 |
| QM | knn | 69.7 [65.0, 74.3] | 61.9 [57.1, 66.6] | 63.7 [59.2, 68.3] | 0.65 |
| FS | logreg | 72.0 [66.8, 77.1] | 67.5 [62.9, 71.9] | 66.3 [61.9, 70.7] | 0.79 |
| FS | knn | 71.8 [67.1, 76.3] | 66.9 [62.2, 71.3] | 66.3 [61.6, 70.8] | 0.83 |
| FM | logreg | 72.5 [67.3, 77.6] | 62.1 [57.4, 66.7] | 64.8 [60.5, 69.0] | 0.45 |
| FM | knn | 71.5 [66.7, 76.0] | 63.9 [59.0, 68.5] | 65.9 [61.3, 70.3] | 1.17 |
| SM | logreg | 63.1 [57.4, 68.1] | 64.2 [59.5, 68.9] | 63.8 [59.4, 68.1] | 0.39 |
| SM | knn | 68.8 [64.1, 73.4] | 63.8 [59.1, 68.5] | 64.9 [60.4, 69.5] | 1.07 |
| QFS | logreg | 71.1 [65.8, 76.1] | 65.1 [60.5, 69.7] | 68.2 [64.0, 72.3] | 0.49 |
| QFS | knn | 71.8 [67.3, 76.5] | 59.3 [54.5, 64.3] | 61.5 [57.2, 66.0] | 1.41 |
| QFM | logreg | 71.9 [66.7, 77.0] | 59.3 [54.6, 63.9] | 60.7 [56.3, 64.8] | 0.45 |
| QFM | knn | 72.3 [67.7, 76.7] | 64.7 [59.9, 69.3] | 64.1 [59.7, 68.6] | 0.93 |
| QSM | logreg | 62.9 [57.3, 67.7] | 61.9 [57.0, 66.9] | 60.7 [56.4, 65.1] | 0.75 |
| QSM | knn | 69.9 [65.1, 74.4] | 63.3 [58.5, 67.9] | 65.6 [61.0, 70.1] | 0.91 |
| FSM | logreg | 72.1 [67.0, 77.1] | 64.4 [60.0, 69.0] | 63.7 [59.4, 67.9] | 0.49 |
| FSM | knn | 70.9 [66.3, 75.4] | 63.9 [59.1, 68.6] | 65.3 [60.8, 69.5] | 1.33 |
| QFSM | logreg | 70.7 [65.5, 75.7] | 59.5 [54.7, 64.0] | 62.7 [58.1, 67.2] | 0.33 |
| QFSM | knn | 71.8 [67.1, 76.3] | 62.2 [57.5, 67.1] | 61.7 [57.3, 66.2] | 1.89 |

### Main effects and two-way interactions

Effects are in percentage points on accuracy, averaged over items and over
matched cells in the full 16-cell layer factorial. Holm p is adjusted across
the 19 terms in this table.

| Term | Effect (pp) [95% CI] | p (sign-flip) | Holm p | N |
|---|---|---|---|---|
| layer Q (main) | -0.6 [-1.5, +0.3] | 0.2058 | 1.0000 | 300 |
| layer F (main) | +2.2 [+0.8, +3.6] | 0.0014 | 0.0238 | 300 |
| layer S (main) | -0.0 [-0.8, +0.8] | 0.9866 | 1.0000 | 300 |
| layer M (main) | -1.6 [-2.3, -0.9] | 0.0000 | 0.0000 | 300 |
| native vs text (main) | +5.4 [+1.7, +9.0] | 0.0070 | 0.1050 | 300 |
| both vs text (main) | +0.6 [-0.7, +2.0] | 0.3644 | 1.0000 | 300 |
| both vs native (main) | -4.8 [-8.4, -0.9] | 0.0168 | 0.2016 | 300 |
| kNN vs logistic probe (main) | +1.2 [-1.9, +4.2] | 0.4448 | 1.0000 | 300 |
| layer Q × native-vs-text | +1.3 [-0.5, +3.0] | 0.1356 | 1.0000 | 300 |
| layer F × native-vs-text | +5.7 [+3.2, +8.2] | 0.0000 | 0.0000 | 300 |
| layer S × native-vs-text | -0.9 [-2.6, +0.8] | 0.3024 | 1.0000 | 300 |
| layer M × native-vs-text | +2.1 [+0.8, +3.4] | 0.0026 | 0.0416 | 300 |
| Q × F | -1.9 [-3.3, -0.4] | 0.0134 | 0.1742 | 300 |
| Q × S | -1.9 [-3.3, -0.5] | 0.0114 | 0.1596 | 300 |
| Q × M | +0.1 [-0.8, +0.9] | 0.8472 | 1.0000 | 300 |
| F × S | -1.1 [-2.4, +0.2] | 0.0856 | 0.9416 | 300 |
| F × M | -0.3 [-1.3, +0.7] | 0.5280 | 1.0000 | 300 |
| S × M | +0.6 [-0.4, +1.6] | 0.2142 | 1.0000 | 300 |
| native-vs-text × kNN | +3.3 [-3.6, +10.3] | 0.3550 | 1.0000 | 300 |

### What each factor changed (CALiSol probes only)

- **F (MoLFormer logS and logD, plus RDKit logP and TPSA).** This is the only
  layer with a clear positive main effect after Holm correction: +2.2 pp
  [+0.8, +3.6], Holm p = 0.024. The gain is concentrated in the native arm
  (F × native-vs-text +5.7 pp [+3.2, +8.2], Holm p < 0.001). The best cells
  in the 16-cell design are native F: logistic 72.7% [67.3, 77.7] and kNN
  73.2% [68.5, 77.6]. Writing the same F facts as text helps much less
  (text F logistic 65.3% [60.5, 69.7]). Adding Q, S, or M on top of native F
  does not improve on F alone; native QFSM is 70.7% [65.5, 75.7] (logistic)
  and 71.8% [67.1, 76.3] (kNN).
- **Q (GFN2-xTB HOMO, LUMO, gap, dipole, vertical IP and EA).** No clear main
  effect (-0.6 pp [-1.5, +0.3], Holm p = 1.00). Two-way interactions of Q
  with F or S are slightly negative on the raw test but do not survive Holm
  correction.
- **S (ALPB solvation free energy, dielectric, viscosity).** No clear main
  effect (-0.0 pp [-0.8, +0.8], Holm p = 1.00).
- **M (melting and boiling points, temperature above the solvent melting
  point).** A negative main effect: -1.6 pp [-2.3, -0.9], Holm p < 0.001.
  The drop is largest when M is written as text (text M logistic 57.9%
  [53.1, 62.9], below the none / text / logistic cell). The M × native-vs-text
  interaction is +2.1 pp [+0.8, +3.4], Holm p = 0.042: native numbers absorb
  M better than text does, but native M still does not beat native none.
- **Native vs text.** Native numbers are ahead on average (+5.4 pp [+1.7,
  +9.0], p = 0.007). That contrast does not survive Holm correction across
  the 19 terms in the table (Holm p = 0.105). The advantage is clearly
  established for layer F, and is also present for layer M, rather than a
  uniform rule for every layer. Concatenating text and native ("both") does
  not beat native alone (both vs native -4.8 pp, Holm p = 0.20).
- **Probe choice.** kNN and logistic regression do not differ reliably.

These conclusions apply to a pairwise conductivity ranking on held-out
CALiSol systems, with linear or kNN probes, not to a language model and not
yet to protocol QA or error correction.

## Other planned cells

| Cell | Status |
|---|---|
| CALiSol layer × modality × probe factorial (16 × 3 × 2, five seeds) | measured, table above |
| Qwen2.5-0.5B-Instruct on ChemProto-Twins and CALiSol (all layer and representation conditions) | **not run** |
| Larger CLI model on ChemProto-Twins (A, B, +QFSM, ALL-shuf, ALL-plac, R2, R3, R4) | **not run** |
| LabLean checker alone as an error detector on expressible twins | **not run** |
| Soft-token arms N0, N1, N2, N3, N2-shuf on CALiSol with a frozen 0.5B LM | **not run** |
| LAB-Bench ProtocolQA and BioProBench ERR, items with at least one resolved small molecule | **not run** |

## How to run

From `ablation/protocol_factors/`:

```bash
# unit tests (from the repo root; CPU, no model download)
python -m pytest -q tests/test_protocol_factors.py

# rebuild ChemProto-Twins from the reviewed scenarios
python assemble_own.py data/scenarios_own.json

# rebuild CALiSol pairs (needs CALISOL_DIR with calisol23.csv and calisol_smiles.csv)
python calisol_pairs.py

# precompute xTB and MoLFormer caches, then run or resume the CALiSol probes
python build_cache.py data/items_own.jsonl data/calisol_train.jsonl data/calisol_test.jsonl
python probe_calisol.py

# rebuild the result tables from whatever result files exist
python report.py
```

Qwen and CLI readers, formal conversions, and the soft-token arms have
scripts in this folder (`run_qwen.py`, `run_llm.py`, `convert_reps.py`,
`n2_calisol.py`). Public benches are prepared with `prep_public.py`, which
writes only item ids and resolved molecule names.

## Licenses and limits

- ChemProto-Twins is a synthetic, small, hand-designed set. It is not a
  substitute for a public protocol benchmark.
- CALiSol-23 derived pairs are CC-BY 4.0. Each pair keeps the source DOI.
  The original dataset is not redistributed here; only the derived pair files
  used in this run are.
- LAB-Bench ProtocolQA is CC-BY-SA 4.0. BioProBench is CC-BY-NC 4.0 and is
  for research use only. Neither source text nor answers are stored in this
  repository.
- MoLFormer-XL weights are Apache-2.0 and are downloaded at run time, not
  stored here.
- xTB numbers are single-conformer GFN2-xTB (MMFF or UFF geometry). They are
  descriptors, not refined first-principles targets.
- All language-model readers, all formal representation conditions, the
  soft-token arms, and the public protocol benches are not run in this
  report. Do not treat the CALiSol probe result as an answer to the protocol
  QA, error-correction, or planning questions until those cells are measured.

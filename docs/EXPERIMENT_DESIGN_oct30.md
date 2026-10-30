# Oct-30: geometry-conditioned molecular language models

Implementation status: code written and statically reviewed; no Python execution, training,
test results, or hardware measurements are claimed in this change. All unmeasured A100
quantities below remain **TBD (A100)**. The CPU smoke driver has a 38-minute deadline;
its successful completion within that deadline still needs to be measured on eight cores.

## Question and experiment matrix

Does an O(3)-invariant interface to equivariant atomic features improve numeric molecular
prediction and transfer to public electrolyte/battery data under an equal processed-token cap?

| Stage | Arms / treatments | Seeds | Primary comparison |
|---|---|---|---|
| QM9 continued pretraining | smiles, smiles_xyz, prefixA, xattnA, xattnB, xattnC | 0, 1, 2 | Each geometry arm vs smiles |
| Answer-only SFT | All six pretrained arms | 0, 1, 2 | Before/after SFT, paired within seed |
| Synthetic numeric DPO | Each corresponding SFT arm, fixed frozen SFT reference | 0, 1, 2 | SFT vs SFT+DPO |
| Frozen downstream probe | Each checkpoint; hidden-state mean pool + ridge | 0, 1, 2 | Geometry vs smiles |
| Full downstream fine-tuning | Each checkpoint + scalar regression head, 3 epochs initially | 0, 1, 2 | Same initial checkpoint and split |
| Molecular baselines | Fraction-weighted Morgan fingerprints + conditions; ridge and RF | 0, 1, 2 | Compare to LM probes/FT |

Repeat downstream evaluation for pretraining-only, SFT, and DPO checkpoints in separate
output directories. The evaluator accepts one checkpoint per arm per invocation. Main
post-training uses QM9 QA only, so downstream holdouts remain independent. Optional CALiSol
or RedDB SFT must use the same downstream split and subset; the evaluator rejects
intersections between task SFT training row IDs and downstream validation/test IDs.
For a rigorous task-specific SFT comparison, also inspect formulation/scaffold overlap,
not just IDs. Independent pretraining-seed replications and downstream-split replications
are separate experimental axes; do not describe three splits of one checkpoint as three
independent pretraining runs.

## Architecture and base model

The primary A100 model is `hf:Qwen/Qwen2.5-0.5B`, continued from its base checkpoint,
with full parameter fine-tuning and no LoRA. This offers a pretrained numeric/text prior
at a tractable scale. Its official model card lists 0.49B parameters and Apache-2.0
licensing; preserve the exact downloaded revision in experiment records.
[Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B).

A separate from-scratch experiment uses `small`: GPT-2-shaped 12 layers, hidden width
768, 12 heads, with this repository's fixed SMILES/character vocabulary. This controls
for unknown molecular content in public LM pretraining. Do not compare perplexities
between Qwen's tokenizer and the character tokenizer. `tiny` is a development model
(4 layers, width 128); smoke overrides it to 2 layers and width 64.

Text arms contain only the decoder LM. `prefixA` converts each atom's parity-correct
invariants into a soft prefix embedding. A/B/C use the corresponding existing
IrrepsToText implementation in an adapter after the middle decoder block. The adapter
adds `tanh(gate) * projection(invariants(cross_attention(hidden, atom_irreps)))` to hidden
states. Gates start at zero. Tests explicitly set gates nonzero for geometry-invariance
checks; a separate test compares zero-gate logits exactly with a shared text LM.
Option C uses smaller cross irreps because its Clebsch–Gordan self-product is expensive.
Parameter counts are therefore reported by LM, encoder, adapters, prefix projector and total.

Mixtures concatenate species graphs with gaps greater than the 9.45-bohr cutoff.
An invariant role embedding (salt/solvent/molecule) and a scalar amount projection enter
the encoder's even scalar input. Arbitrary species IDs are deliberately omitted to
avoid dependence on listing order. The amount is the normalized **recorded ratio** for
solvents, and recorded concentration for salt. It is a mole fraction only when the
source ratio is molar. Prompts retain mass/volume/molar basis and concentration units;
no unavailable density conversion is assumed. Disconnected salt fragments are treated
as separated ions, with no contact ion pairs, solvation shell or intermolecular edges.

## Data, provenance and licensing

Only public sources are used. No proprietary or SES AI records are included. Raw data,
conformer caches, MP structures and model weights remain outside version control.
Dataset license terms are distinct from article and repository-code licenses.

| Source and citation | Local inputs / targets | Access and license provenance |
|---|---|---|
| [QM9, Ramakrishnan et al. 2014](https://doi.org/10.1038/sdata.2014.22), [Figshare record](https://doi.org/10.6084/m9.figshare.978904) | `gdb9.sdf`, locally generated `qm9_xtb.pkl`; energy (hartree), dipole (e·bohr), norm | Record the Figshare data license with the downloaded snapshot; do not substitute the article's CC BY-NC-SA 4.0 license for the data license. xTB labels are local derived calculations. |
| [CALiSol-23, de Blasio et al. 2024](https://doi.org/10.1038/s41597-024-03575-8), [source repository](https://github.com/Pele0599/CALiSol-23) | Conductivity k (mS/cm), temperature, concentration, salt, 38 solvent fractions; `log10(k+0.001)` | Article CC BY 4.0; source CSV redistribution license must be recorded from the upstream dataset snapshot before redistributing raw records. This repo supplies a downloader, not a new data license. |
| [RedDB, Sorkun et al. 2022](https://doi.org/10.1038/s41597-022-01832-2), [Harvard Dataverse](https://doi.org/10.7910/DVN/F3QFSQ) | `reaction_energy`, `reactant_solubility`, `reactant_lumo`, `reactant_homo` | Article CC BY 4.0; capture the Dataverse dataset's license/terms with the downloaded version. License metadata could not be verified from Dataverse during this implementation; do not assert a waiver. |
| [Materials Project, Jain et al. 2013](https://doi.org/10.1063/1.4812323), [official API](https://docs.materialsproject.org/downloading-data/using-the-api) | Insertion-electrode average voltage (V), gravimetric capacity (mAh/g), gravimetric energy (Wh/kg) | API key required; follow MP data terms and per-record provenance. Record database release and applicable license before redistribution. Skip cleanly when `MP_API_KEY` is absent. |

`scripts/download_public_data.py` fetches the three electrolyte files listed in the
spec, writes a URL/size/SHA-256 manifest, and prints checksums. `--expected` accepts
trusted hashes for verification. Without expected hashes these are provenance hashes,
not independent authenticity checks. QM9 SDF and xTB label generation are separate
prerequisites; the downloader does not pretend the experimental xTB pickle is public.

`gdb9.sdf.csv` has no SMILES. The corpus derives canonical heavy-atom SMILES from the
SDF supplier's **zero-based index**, matching `01_make_labels.py`. Invalid SDF entries
retain their original indices and are skipped without reindexing later molecules.
The geometry is the noisy xTB-input geometry in bohr, not regenerated RDKit geometry.
Text includes SMILES, energy, dipole vector and norm; `smiles_xyz` additionally writes
Angstrom coordinates to two decimals. Adapter/prefix arms receive the SMILES corpus
plus geometry, without xyz text.

CALiSol mapping accepts both two-field solvent rows and three-field rows and handles
literal `\\r\\n` separators in the downloaded mapping. EC/PC/DMC/EMC aliases are explicit.
Invalid/unmapped species are logged; affected rows are excluded. Nonfinite targets,
negative conductivity/concentration/fractions, invalid temperatures and zero-total
fractions are dropped. Remaining solvent fractions are normalized within recorded
basis. Source SMILES, including salt chemistry, still require a chemical quality audit.
The fingerprint baseline weights solvent fingerprints by these fractions and salt by
concentration, then appends T, c, ratio basis and concentration-unit indicators.

RedDB orbital energies and reaction energies are retained in source atomic units;
solubility is retained on the source's provided scale. These are computed proxies,
not measured electrolyte stability windows. A reactant-only redox regression can be
ambiguous if one reactant appears in different reactions; repeated reactants remain
together in every split. Audit reaction definitions and source convergence flags
before drawing chemical conclusions. Missing labels are filtered per target after
splitting, preserving sample membership for other targets.

RDKit conformers use deterministic ETKDGv3 + MMFF94, then UFF or raw ETKDG fallback;
one thread per embedding and a persistent SMILES/seed cache. Conformer failures are
excluded identically for every arm, including fingerprint baselines.

For MP, replicate enough cells to cover a 6-Angstrom sphere about the original cell
centre, then keep the nearest at most 64 atoms. Atomic numbers >=100 and disordered
sites are unsupported. This is an **unpassivated finite-cluster approximation**:
periodic edges, surface physics and long-range electrostatics are absent. Use the
charged endpoint structure without target values in prompt text. Molecular Morgan
fingerprints have no crystal interpretation, so MP omits that baseline explicitly.

## Splits and preventing leakage

QM9 membership is determined once from sorted unique `qm9_index` values with seeded
80/10/10 assignment; all arms save the same molecule IDs. The corpus subset is chosen
before arm rendering. Do not change subset size between arms or resumed sessions.

CALiSol primary split holds out whole salt + solvent-set formulation groups. Secondary
splits are random rows and leave-DOI-out (`--split doi`), with the latter checking
publication-specific correlations. Formulation grouping ignores concentration and
ratios intentionally, so a formulation cannot leak through another temperature.

RedDB primary split groups by RDKit Murcko scaffold; acyclic molecules share the empty
scaffold. Its random control groups by canonical reactant SMILES, retaining duplicate
reactants together. Whole-group splits approximate the target fractions; exact row
counts can differ substantially for large groups. Empty train/validation/test sets
are reported as skipped, never silently replaced by a leaking random split.
MP groups by charged material ID; related polymorph/composition leakage remains a
threat requiring a stricter family split in a later study.

Fit feature normalization and regression-target normalization on training rows only.
Select ridge alpha, RF leaf size and fine-tuning epoch on validation rows only. Test
labels never select settings. SFT builders receive explicit split subsets. The default
post-training source is QM9; `--sources qm9 reddb calisol` enables task-specific QA.
Synthetic DPO chosen answers are formatted true values and rejected answers are seeded
perturbations, with answer-only log-probability sums and a frozen initial reference.
This tests synthetic numeric preferences, not human preference alignment.

## Token budget and compute accounting

For each stage and seed, fix the same `--tokens_budget` across all six arms, tokenizer,
text context cap, batch size, optimizer settings and data order. Count nonpadding text
tokens processed, including BOS/EOS and masked prompt tokens. Prefix arms also count
valid atom prefix tokens. SFT loss masks every token except the numeric answer; its
budget still includes prompts. DPO counts both answers' policy and frozen-reference
forward tokens, including prefixes. DPO and SFT budgets must be reported separately
because their backward-pass costs differ.

The trainer never exceeds the cap. The final LM/SFT batch may shorten text to fit;
if fewer than prefix+two text tokens fit, or a truncated SFT tail has no answer, it
stops below the cap. DPO requires complete preference pairs and can leave less than
one pair's cost unused. Every run reports the actual token count and shortfall.
Compare actual counts; increase the common cap if the relative shortfall is material.
Masked padded slots are excluded from accounting even though dense kernels still do
work on them. Equal processed tokens are a **compute proxy**, not exact equal FLOPs.

The pretraining FLOPs estimate is `6 * total_parameters * tokens_seen`; it excludes
quadratic attention and geometry/CG operations. Report wall time, throughput, parameter
counts, GPU utilization and peak allocated memory alongside this proxy. Do not call
this an exact matched-FLOPs experiment. Final A100 budgets are TBD (A100); choose a
common token cap after profiling the slowest arm on training data only.

Default contexts are at least 1024 positions for scratch LMs. Downstream evaluation
reserves prefix space and applies one common text limit across arms. Overlong inputs
are right truncated for LM losses/features; numeric decoding skips prompts that would
be truncated and reports their count. SFT raises if an answer is entirely truncated.
Report lost-property-token coverage in xyz arms; truncation can otherwise dominate
an apparent geometry advantage.

## Measurement and statistical reporting

Report validation next-token NLL/perplexity, with separate SMILES, numeric-property,
xyz and other token groups. Numeric MAE uses greedy decoded values and reports parse
success and attempted counts; MAE conditional on successful parsing is insufficient
without coverage. Numeric pretraining queries use the preceding corpus text, so later
properties may condition on earlier true properties. SFT numeric QA has no other
property answers in context. The printed targets are rounded; numeric evaluation
cannot distinguish precision below that formatting scale.

For downstream tasks report MAE, RMSE, R2 and tie-aware Spearman. CALiSol additionally
reports original mS/cm errors after inversion of `log10(k+0.001)`, clipped at zero.
Save per-row predictions and IDs and per-seed 95% test bootstrap intervals. Report
mean metrics across three seeds. Compare each arm with smiles by a hierarchical
paired bootstrap: resample seeds, then the same test rows for both arms within each
seed, and estimate MAE differences with 95% intervals. Default 1000 bootstrap draws;
smoke uses 200. The nonnegative-difference bootstrap mass is descriptive, not an
exact frequentist p-value. Also pre-register an across-seed paired permutation test
on seed mean MAE differences for final reporting; with only three seeds, significance
resolution is limited. Correct exploratory comparisons for multiplicity. Correlated
rows within publications/formulations/scaffolds motivate a future cluster bootstrap;
row bootstrap intervals should be interpreted conditionally, not as independent-study
uncertainty.

## A100 sessions and reproducibility

Treat a Colab session as approximately 12–24 hours of available time, subject to actual
runtime limits. Plan each notebook segment for at most four hours, retaining time for
evaluation and upload. These are planning constraints, not measured guarantees.

| Segment | Content | Token cap / runtime / memory |
|---|---|---|
| Pilot | One short run per arm, warmup and bounded trace | TBD (A100) |
| Pretraining | One arm/seed per segment, split further if needed | TBD (A100) |
| Post-training | SFT, then DPO from SFT in separate segments | TBD (A100) |
| Transfer | Frozen probes, then task-specific full fine-tuning | TBD (A100) |

Checkpoints include model, optimizer, Python/NumPy/Torch/CUDA RNG, epoch permutation
seed, batch cursor, steps and tokens. Writes use an atomic latest marker; two recent
checkpoints are retained. Resume with identical architecture, split, cap and training
settings. `--stop_after_steps` provides controlled segment boundaries without changing
the original LR schedule/token cap. Copy the complete run directory, including config
and `ckpt/`, to persistent storage between notebook sessions. Do not overwrite the
source checkpoint while post-training; DPO needs its original frozen reference.

Profiling writes `summary.json` with data/forward/backward/optimizer timings, samples/s,
tokens/s, CPU RSS and CUDA peak bytes, plus optional NVML/nvidia-smi utilization samples.
CUDA timings synchronize boundaries. A bounded torch.profiler trace and operator table
cover the first few steps. Profiling itself affects timings; collect a warm steady-state
run separately for publication-quality throughput. Peak CPU RSS is available through
the resource fallback; psutil reports current RSS. Evaluation throughput is not included
in pretraining training-phase summaries.

## Commands and outputs

Run from the repository root after installing the existing environment plus `pandas`;
`scikit-learn`, `psutil` and `pynvml` are optional. MP additionally needs `mp-api` and
`pymatgen`. No network is needed for scratch-model tests. HF runs download their base
model/tokenizer on first use; pin/cache these externally for reproducibility.

```bash
python scripts/download_public_data.py
python scripts/20_pretrain_lm.py --arm xattnA --model tiny --tokens_budget 1000000 --out runs/oct30/xattnA_s0 --seed 0 --device cuda
python scripts/20_pretrain_lm.py --arm xattnA --model tiny --tokens_budget 1000000 --out runs/oct30/xattnA_s0 --seed 0 --device cuda --resume
python scripts/21_posttrain.py --pretrained runs/oct30/xattnA_s0 --out runs/oct30/sft_xattnA_s0 --tokens_budget 100000 --device cuda
python scripts/21_posttrain.py --pretrained runs/oct30/sft_xattnA_s0 --out runs/oct30/dpo_xattnA_s0 --dpo --tokens_budget 100000 --device cuda
python scripts/22_downstream_eval.py --runs runs/oct30/smiles_s0 runs/oct30/xattnA_s0 --tasks calisol reddb --seeds 0 1 2 --out results/oct30_main --device cuda
python scripts/23_smoke_cpu.py
pytest -q tests/test_ylm_lm.py
```

The numeric token caps in these commands are examples, not tuned A100 budgets.
Smoke writes under `results/oct30_smoke/`: all six pretraining runs, short SFT for smiles
and xattnA, three-seed group/scaffold probes for all six arms and fingerprint baselines,
per-stage logs, JSON outcomes and `SUMMARY.md`. Timeout or subprocess failure produces
a failed summary and a nonzero exit status. No successful outcome is pre-populated.
Smoke does not run expensive full downstream fine-tuning or DPO.

## Threats to validity

Adapters add parameters and geometry compute; gates at zero only match the initial
text logits. Prefix tokens change positions and attention cost. Smaller C cross irreps
make capacity unequal. XYZ text leaks rounded, frame-dependent geometry through a
longer sequence, while equivariant pathways are invariant. Dipole components in the
pretraining text are frame-dependent targets; invariant adapters cannot recover an
arbitrary frame from coordinates alone. Dipole norm/energy are the relevant invariant
numeric comparisons; component loss invariance with fixed labels is a representation
check, not an ability to predict rotated vector components.

RDKit conformers are neither DFT structures nor electrolyte configurations. Dissociated
salts, recorded nonmolar fractions and finite crystal clusters are approximations.
Source mapping mistakes, digitized conductivity uncertainty, noisy xTB positions,
reaction-label ambiguity and scaffold imbalance can affect conclusions. Shared public
LM pretraining may already contain benchmark molecules. Row bootstrapping understates
publication-level dependence. CPU smoke throughput cannot predict A100 performance.
Track these limitations and retain all measured failure/parse rates alongside scores.

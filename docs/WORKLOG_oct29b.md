# Worklog: October 29, options B/C and parity re-evaluation

## Scope and local availability

Read README.md, 03_technical_design.md, progress.md, pipeline scripts 01–04,
checkpoint evaluation/benchmark code, model/encoder/attention, helpers, B/C-lite,
and tests. No git commands were run. Python was `/workspace/venv311/bin/python`;
CPU only, two PyTorch/OpenMP/MKL threads per run. No agents were spawned.

A recursive `/workspace` file search found three actual released YLM checkpoints:
`checkpoints/v0/{baseline_taskid_s0,ylm_minilm_s0,ylm_matscibert_s0}.pt`.
There were **no YLM labels, question files, cached text embeddings, QM9 SDF, or
local trained runs**. Other repositories' unrelated checkpoints and RDKit example
SDFs are not TensorQA data. The checkpoints contain weights/scales/configuration,
not labels or frozen text encoders.

Missing for real accuracy/training: `data/gdb9.sdf`, `data/qm9_xtb.pkl`,
`data/tensorqa_v0.json`, `data/text_minilm.pt` (or MatSciBERT equivalent),
`tblite` and `transformers` (both absent). RDKit, torch, numpy and e3nn are installed.
`nbformat` is absent, so notebook validation used JSON plus compilation as allowed.

Dependency/download attempts:

```bash
uv pip install -p /workspace/venv311/bin/python tblite transformers nbformat
# Failed: default /home/box/.cache/uv is read-only.
UV_CACHE_DIR=/tmp/ylm-uv uv pip install -p /workspace/venv311/bin/python tblite transformers nbformat
# Failed after retries: pypi.org DNS lookup failure.
curl -I --connect-timeout 10 --max-time 20 https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip
# Failed: could not resolve DeepChem host.
```

**No real chemical training, MAE or cosine evaluation was possible locally.**
No new molecular MAE/cosine numbers are claimed and no historical v0 accuracy
numbers are relabeled as post-fix results. Retraining old/fixed keys for minutes
would not solve the missing labels/dependencies. The notebook supplies that work
for a connected Colab environment.

## Implemented behavior

- `YLM(..., option="A"|"B"|"C", legacy_odd_scalars=False)`; A remains default
  and loads the three old checkpoints strictly without parameter renaming.
- B: masked global mean irreps, per-token text gates on global channels,
  equivariant atom-key projection, dot products of matching `(l, parity)`
  channels added to A's score before atom softmax. Matching parity is essential
  for reflection invariance.
- C: A output followed by `FullyConnectedTensorProduct(o_t, o_t, MLP(h_t))`,
  with external, unshared, text-generated path weights. This is the full CG
  construction from the design, not the old compositional B/C-lite model.
  A dedicated vector-only-input test verifies nonzero generated `2e` output
  through `1o x 1o -> 2e`. The full YLM answer head still returns a polar vector.
- `legacy_odd_scalars=True` reproduces raw odd-scalar keys, without changing
  parameter shapes. It deliberately loses O(3) guarantees.
- Script 04: option/legacy/tiny flags, bounded evaluation batch size,
  rotation-only, combined O(3), and reflection-only metrics, portable
  `checkpoint.pt` with architecture args and scales in addition to `model.pt`.
- Script 09 reloads tiny/A/B/C and legacy architectures and supports reflection
  evaluation. Training/evaluation import the existing older-e3nn compatibility
  initialization before importing e3nn under current PyTorch.
- Added isolated-process full-model CPU benchmark, released-checkpoint symmetry
  diagnostic, and self-contained Colab notebook. Updated README/design notes.

## Released checkpoint re-evaluation: symmetry only

Executed `scripts/check_checkpoint_parity.py` on one eight-atom synthetic geometry
per checkpoint, float32, seed 29, two threads, force task. MiniLM/MatSciBERT use
random token states because real frozen text states are missing. Task-ID uses its
learned token. Same geometry/transform is used for old/fixed comparisons within
each checkpoint; different checkpoints receive different seeded draws.
This is a symmetry diagnostic, **not a labeled accuracy benchmark**. Near-identical
old/fixed results for the two language checkpoints are limited to these draws;
random text can saturate attention and hide the old parity defect.

| Checkpoint | Keys | Rotation relative error | Reflection relative error |
|---|---|---:|---:|
| baseline_taskid_s0.pt | raw odd (old) | 4.6e-07 | 0.083 |
| baseline_taskid_s0.pt | abs odd (fixed) | 3.5e-07 | 8.66e-07 |
| ylm_matscibert_s0.pt | raw odd (old) | 3.61e-07 | 4.36e-07 |
| ylm_matscibert_s0.pt | abs odd (fixed) | 3.61e-07 | 4.36e-07 |
| ylm_minilm_s0.pt | raw odd (old) | 7.98e-07 | 2.75e-06 |
| ylm_minilm_s0.pt | abs odd (fixed) | 7.98e-07 | 2.75e-06 |

All raw absolute/relative and attention errors are saved in
`docs/checkpoint_parity_oct29b.json`. The task-ID draw exposes the reflection
failure clearly; its attention max difference drops from 0.1005 to 7.15e-7.
These are existing weights evaluated with different forward behavior, not retrained weights.

## Measured full-model CPU benchmark

Encoder + cross-attention + token pooling + vector head, float32, batch 1,
text input dimension 32, synthetic geometries, two threads. The frozen Hugging
Face text encoder is excluded: its states are inputs, as in training. Timings
exclude initialization and use 3 warmups/20 iterations for default widths.
Backward-only timings build the loss graph outside the measured interval;
forward/backward timings include loss construction and gradient clearing, but
no optimizer step. RSS is Linux isolated child-process peak resident memory,
including Python/PyTorch, model initialization, compiled e3nn code and graphs;
it is **not** a tensor-allocation measurement. Every grid case runs in a fresh
process so peaks do not accumulate between rows.

| Option | Atoms | Tokens | Parameters | Forward ms | Backward ms | Forward+backward ms | Peak RSS MiB | Reflection rel. error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 8 | 4 | 250,369 | 4.54 | 8.62 | 13.89 | 340.8 | 1.13e-06 |
| A | 8 | 16 | 250,369 | 3.85 | 9.08 | 14.11 | 340.7 | 2.17e-06 |
| A | 24 | 4 | 250,369 | 7.31 | 20.26 | 28.87 | 374.7 | 7.83e-07 |
| A | 24 | 16 | 250,369 | 7.43 | 20.49 | 32.31 | 375.8 | 1.38e-06 |
| B | 8 | 4 | 288,769 | 6.22 | 9.43 | 15.24 | 342.2 | 6.46e-07 |
| B | 8 | 16 | 288,769 | 4.02 | 9.57 | 14.70 | 342.3 | 2.06e-07 |
| B | 24 | 4 | 288,769 | 9.58 | 18.65 | 29.18 | 376.2 | 6.61e-07 |
| B | 24 | 16 | 288,769 | 7.66 | 19.07 | 27.65 | 373.9 | 3.62e-07 |
| C | 8 | 4 | 13,046,337 | 6.46 | 26.48 | 37.21 | 455.9 | 1.31e-07 |
| C | 8 | 16 | 13,046,337 | 8.58 | 51.63 | 61.02 | 500.2 | 1.68e-06 |
| C | 24 | 4 | 13,046,337 | 10.33 | 41.83 | 57.74 | 479.5 | 1.92e-06 |
| C | 24 | 16 | 13,046,337 | 12.00 | 62.76 | 78.45 | 523.4 | 1.1e-06 |

Shared-host short timings are noisy; small A/B differences do not demonstrate a
speed advantage. C's full hypernetwork is much larger than A/B, especially at
large channel multiplicity. Peak observed RSS stayed below 524 MiB per case.
Full rotation and reflection errors and version metadata are in
`docs/bench_full_model_default_oct29b.json`.

The tiny three-layer preset was also measured on the same 2x2 atom/token grid
(2 warmups/5 iterations): A/B/C have 55,413 / 60,131 / 111,381 parameters at text
dimension 32. Measured forward/backward spans 12.42–23.95 ms, RSS 335.7–347.1 MiB.
Full rows are in `docs/bench_full_model_oct29b.json`.

Commands executed (the default benchmark was first smoke-run with 3 iterations,
then replaced by the 20-iteration measurements above):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /workspace/venv311/bin/python scripts/check_checkpoint_parity.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /workspace/venv311/bin/python scripts/bench_full_model.py --output docs/bench_full_model_oct29b.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /workspace/venv311/bin/python scripts/bench_full_model.py --preset default --iterations 20 --warmup 3 --output docs/bench_full_model_default_oct29b.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /workspace/venv311/bin/python -m pytest -q
```

## Validation

The full suite passes: **25 tests**. Tests cover double-precision proper and
improper rotations, output parity, invariant attention, atom permutation,
padding of atoms/text, translation, and backward gradients for A/B/C, plus
B's relative-direction sensitivity and C's generation of a quadrupole from
vectors with gradients through text-generated CG weights.

Also ran script 04 end to end for **two optimizer steps of tiny C** using a
20-molecule **random-label integration fixture** under `/tmp/ylm_pipeline_smoke`,
then loaded its portable checkpoint with script 09 and evaluated proper/improper
transforms. This verifies training/save/reload/evaluation plumbing only; random
labels have no physical meaning and its MAE/cosine values are not scientific
results. It was not a QM9-like labeling/training experiment.

```bash
# cwd /tmp/ylm_pipeline_smoke; fixture generated with numpy/torch, seed 29
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /workspace/venv311/bin/python /workspace/repos/YLM/scripts/04_train.py --text minilm --option C --tiny --name smoke_C --epochs 1 --max_steps 2 --batch_size 12 --eval_rotations 2 --eval_reflections --eval_batch_size 4 --device cpu
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /workspace/venv311/bin/python /workspace/repos/YLM/scripts/09_eval_checkpoint.py --ckpt /tmp/ylm_pipeline_smoke/runs/smoke_C/checkpoint.pt --data_dir /tmp/ylm_pipeline_smoke/data --eval_rotations 2 --reflections --batch_size 4
```

All source/scripts parse. `colab/ylm_train.ipynb` parses as nbformat-4 JSON and
every Python code cell compiles. `nbformat` itself could not be installed due
to DNS failure. Colab cells were not executed locally; Drive/GPU/network execution
remains to be verified on Colab.

## Exactly what remains to run on Colab

Open `colab/ylm_train.ipynb`, choose a GPU runtime, and run all cells:

1. Mount Drive; clone `https://github.com/yaniguan/YLM.git`, checkout
   `feat/options-bc-oct29`, install torch-compatible e3nn, RDKit, tblite,
   transformers and other dependencies. The parent must publish this branch's
   changes before the notebook can fetch them.
2. Download QM9, run script 01 with 500 molecules/two tblite workers, script 02
   for molecule-disjoint questions and script 03 for frozen MiniLM states.
3. Run pytest, then script 04 on tiny A/B/C, task-ID baseline and legacy A,
   all seed 0, batch 16, at most 400 steps over at most 20 epochs. Task-ID is the
   baseline here; Qwen/pipeline/scalar training is outside this notebook.
4. Record MAE/cosine/equivariance on normal/held-out wording test splits,
   rotations and reflection-only transforms; reload all portable checkpoints
   with script 09 for independent combined-O(3) evaluation. These are the
   missing real post-parity-fix accuracy measurements. Short runs are exploratory;
   increase steps/epochs and seeds before comparing model quality.
5. Save data/configuration/checkpoints/logs/results directly beneath
   `/content/drive/MyDrive/YLM_runs/<timestamp>/`, run the CPU benchmark and
   zip all outputs in the final cell for download (also retained on Drive).

The notebook labels its broad T4/L4/A100 runtime ranges as estimates grounded
in measured tiny CPU steps and the repo's documented ~35-minute V100 full run.
No Colab/GPU speed measurements are claimed.

## Addendum (parent agent, network available): real post-parity-fix re-evaluation

Data regenerated on the box with network: DeepChem `qm9.zip` → `scripts/01_make_labels.py` (20,000 molecules, tblite GFN2-xTB, 16 s with 7 workers), `02_make_questions.py` (84,787 questions, 571 unique strings, 156 molecules skipped for huge forces), `03_embed_text.py --name minilm`. Released v0 checkpoints evaluated with the parity-fixed code:

```bash
OMP_NUM_THREADS=3 python scripts/09_eval_checkpoint.py --ckpt checkpoints/v0/<ckpt>.pt --splits test test_heldout --eval_rotations 4 --reflections --batch_size 64
```

| checkpoint | split | force MAE | force cos | atom acc | dipole MAE | dipole cos | eq_err max |
|---|---|---|---|---|---|---|---|
| ylm_minilm_s0 (fixed keys) | test | 0.00727 | 0.9446 | 0.754 | 0.0846 | 0.9792 | 2.9e-4 |
| ylm_minilm_s0 (fixed keys) | test_heldout | 0.01312 | 0.8764 | 0.706 | 0.2664 | 0.8171 | 2.6e-4 |
| baseline_taskid_s0 (fixed keys) | test | 0.04209 | 0.117 | 0.011 | 0.0724 | 0.9850 | 7.3e-5 |

Released `results.json` (pre-fix, original data) for ylm_minilm_s0 test: force MAE 0.00745, cos 0.9445, atom acc 0.746; dipole MAE 0.0830. So the parity fix does not degrade the trained MiniLM model (the regenerated labels may differ slightly from the original data, so small differences are not attributable to the fix). Wall time 285 s / 315 s on CPU. Logs: `results/oct29b/`. Retraining A/B/C is left to `colab/ylm_train.ipynb`.

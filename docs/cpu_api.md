# CPU inference API: measured latency and accuracy

`src/api.py` answers one TensorQA-v0 question (dipole or per-atom force) about one molecule.
`scripts/bench_api.py` produced every number below; raw output: [`results/cpu_api/bench.json`](../results/cpu_api/bench.json).

## Setup

- CPU: `Intel(R) Xeon(R) Processor` (8 logical CPUs), 1 torch thread, torch 2.15 (CPU build).
- The machine was shared with other jobs (load average 15–18 during the run). **Wall** time includes waiting for the CPU; **CPU** time is the process time of the request, i.e. its cost on an idle core.
- 600 questions: 300 from `test` and 300 from `test_heldout` (seed 0) of the `tensorqa_v0.tar.gz` release asset. Inputs are atoms + positions (bohr).
- "Cold" = the question-embedding cache is cleared before every request, so the text encoder always runs. Model loading and warm-up are not included.

## Latency per request (ms)

| Variant | cold wall p50 / p95 | cold CPU p50 / p95 | warm wall p50 / p95 | SMILES input, first call, wall p50 / p95 |
|---|---|---|---|---|
| **MatSciBERT fp32 (default, the released model)** | 112 / 206 | 79 / 105 | 11 / 103 | 110 / 209 |
| MatSciBERT, int8 text encoder | 39 / 61 | 35 / 43 | 11 / 55 | 39 / 58 |
| MiniLM fp32 | 20 / 42 | 18 / 22 | 12 / 28 | 24 / 47 |
| MiniLM, int8 text encoder | 14 / 24 | 13 / 17 | 11 / 22 | 19 / 41 |

All variants are under 1 s per request. The default keeps the full released model, so the 1 s target needs no accuracy trade-off.

## Accuracy (same 600 questions, xTB labels)

| Variant | force cos test / held-out | atom hit test / held-out | dipole cos test / held-out | agreement with default (force cos, held-out) |
|---|---|---|---|---|
| **MatSciBERT fp32** | 0.946 / 0.910 | 0.946 / 0.929 | 0.974 / 0.907 | — |
| MatSciBERT int8 | 0.925 / 0.896 | 0.933 / 0.920 | 0.977 / 0.912 | 0.979 |
| MiniLM fp32 | 0.924 / 0.888 | 0.759 / 0.737 | 0.975 / 0.878 | 0.889 |
| MiniLM int8 | 0.882 / 0.773 | 0.732 / 0.638 | 0.965 / 0.957 | 0.765 |

The force MAE of the default on `test` is 0.0073 hartree/bohr. That is in line with the 0.0068 reported for this checkpoint on the full test split. "Atom hit" means the attention-routed atom is the atom the question describes.

## Scope and limits

- The scope is TensorQA-v0 only: QM9-size H/C/N/O/F molecules (≤ 64 atoms including hydrogens), GFN2-xTB labels, and dipole or per-atom force questions.
- Other questions (for example lab-protocol questions) are rejected. This model does not generate protocols.
- SMILES input is embedded with RDKit ETKDG + MMFF. Training used noised QM9 geometries, so this is a distribution shift.
- The first call after start-up also loads MatSciBERT (about 5 s). Run it as a long-lived process.

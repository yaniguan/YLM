# Options A/B/C — full Colab runs (Oct 30–31, 2026)

Source: `MyDrive/YLM_runs/options_bc_test/runs/full_{A,B,C}/results.json` and `log.txt`, branch
`feat/options-bc-oct30`. All runs: 20,000 QM9 molecules (GFN2-xTB labels), MiniLM text, 60 epochs,
batch 128, lr 1e-3, seed 0, resumable training (`--resume`), eval with 4 rotations + reflections.
Profiling for each run is in `runs/<name>/profiling/` (plus `profiling_session_*` from earlier sessions).

## TEST split (force task) vs targets

| Option | Force MAE (≤ 0.00727) | Force cosine (≥ 0.9446) | Atom acc (≥ 0.754) | All targets |
|---|---|---|---|---|
| A | 0.00668 ✅ | 0.9559 ✅ | 0.909 ✅ | **pass** |
| B | 0.00680 ✅ | 0.9588 ✅ | 0.696 ❌ | fail (atom acc) |
| C | **0.00487** ✅ | **0.9762** ✅ | **0.920** ✅ | **pass** |

Dipole TEST MAE / cosine: A 0.0887 / 0.9781, B 0.0894 / 0.9766, C 0.0735 / 0.9835.

## Held-out wordings (TEST_heldout, force)

| Option | MAE | Cosine | Atom acc |
|---|---|---|---|
| A | 0.0107 | 0.903 | 0.869 |
| B | 0.00952 | 0.923 | 0.666 |
| C | 0.0125 | 0.892 | 0.862 |

C is best in-distribution but generalises to unseen question wordings slightly worse than A on
force MAE/cosine; B is best on held-out MAE/cosine but has the weakest atom accuracy.

## Equivariance

Mean force equivariance error is ~2e-6 for every option on rotations, O(3) (rotations + reflections)
and reflection-only evaluation (A 2.09e-6, B 2.03e-6, C 2.03e-6 reflection-only mean), i.e. float32
round-off. Max errors: A 1.9e-4, B 3.0e-4, C 8.9e-4 (reflection-only).

## Training notes

- Option C has 13,137,857 parameters; ~148 s/epoch on the Colab GPU (from `log.txt`).
- Final full_C validation (epoch 59): force MAE 0.00486, cosine 0.976, atom acc 0.920.
- full_C ran across 6 Colab sessions (resumes at epochs 4, 13, 25, 35, 47); the five interrupted sessions
  lasted ~11–31 min each before the runtime disconnected; the sixth finished (epochs 47–59 + TEST). Per-epoch `latest.pt` checkpoints made every resume lossless
  apart from the interrupted partial epoch. See "Hang/disconnect mitigations" in the PR.

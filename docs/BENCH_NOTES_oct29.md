# October 29 CPU benchmark and development notes

Added a narrow `slice` safe-global registration in `src/__init__.py` and test initialization before e3nn import. This preserves weights-only loading and does not modify site-packages. Standalone callers importing e3nn first must import `src` first on affected e3nn versions.

Fixed O(3) parity in invariant attention keys: odd scalars now use absolute values. Parameter shapes are unchanged; existing checkpoints load, but predictions may change for nonzero odd scalar features. Even-scalar and higher-order norm behavior is unchanged.

Added IrrepsToText and YLM tests for proper/improper rotations, parity, atom permutations, padding, and nonzero finite input/parameter gradients. Cross-attention has independent token outputs (no text mask argument); its padding test checks prefix preservation. YLM tests verify masked token pooling. Existing translation and baseline tests remain. Float64 test defaults are restored after each relevant test. No unavailable heavy dependencies were needed or installed.

Added `collate_molecules` for variable atom/token lengths and `predict_vector` for a molecule and question using a caller-provided text embedding callback. Tests compare batched versus single inference, validate padding and malformed inputs, and check model training-mode restoration. No model weights or tokenizer are downloaded. Coordinates must use checkpoint training units (v0: bohr).

## Commands actually run

```bash
source /workspace/venv311/bin/activate
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/bench_irreps_to_text.py --output docs/bench_oct29.json
```

Final full test run: **13 passed**, 337 upstream e3nn/TorchScript warnings, **9.56 s**. Initial reproduction failed during collection with the unsupported `slice` global.

## Measured benchmark

Synthetic IrrepsToText layer only, CPU, batch 1, float32, two Torch threads, seed 29, 3 warmups and 20 timed iterations per case. Mean wall time in milliseconds. Forward uses no-grad; backward-only constructs the graph outside the timer; forward+backward includes gradient clearing and squared-output mean loss. No optimizer step. One independently sampled proper rotation per case; errors compare full irrep outputs. These small timings contain scheduling noise and are not full-model or trained-checkpoint performance.

Environment: PyTorch 2.10.0+cpu, e3nn 0.4.4. Atom/output irreps `16x0e+4x0o+8x1o+4x1e+4x2e`, text dimension 32, key dimension 16.

| Atoms | Tokens | Forward ms | Backward ms | Forward+backward ms | Max absolute error | Relative L2 error |
|---|---|---|---|---|---|---|
| 8 | 4 | 0.2506736999976056 | 0.6802823499697297 | 1.214582449995305 | 1.9073486328125e-06 | 1.191670335742856e-07 |
| 8 | 16 | 0.2454916000033336 | 0.6862812999656853 | 1.2145806999910747 | 2.86102294921875e-06 | 1.1993554949185636e-07 |
| 8 | 32 | 0.24495600000591367 | 0.7065881000130503 | 1.271180399999139 | 1.1920928955078125e-06 | 1.074809929946241e-07 |
| 32 | 4 | 0.2570197999943957 | 0.6980163999855904 | 1.2671350999994502 | 2.384185791015625e-06 | 1.4198495534856193e-07 |
| 32 | 16 | 0.27136125000879474 | 0.7027557000128581 | 1.2648365500012915 | 5.7220458984375e-06 | 1.548551864516412e-07 |
| 32 | 32 | 0.2664510500039796 | 0.7195604999765237 | 1.3013796499990349 | 7.62939453125e-06 | 1.2399249271766166e-07 |
| 64 | 4 | 0.27464839999993274 | 0.7398578999982419 | 1.3080222999974467 | 4.76837158203125e-06 | 1.256105832680987e-07 |
| 64 | 16 | 0.5284741499963275 | 0.7396396499643743 | 1.3392811500011703 | 5.7220458984375e-06 | 1.4745394594228856e-07 |
| 64 | 32 | 0.2874299000040992 | 0.7736519000104636 | 1.4042535000044154 | 7.62939453125e-06 | 1.833676464002565e-07 |

Raw measured values and configuration are also in `docs/bench_oct29.json`.

## Commit limitation

Attempted logical commits for compatibility/symmetry and helpers, but `git add` failed with `fatal: Unable to create '/workspace/repos/YLM/.git/index.lock': Read-only file system`. No commits were created. The managed environment explicitly mounts `.git` read-only and disables escalation. Suggested commit groups: compatibility plus symmetry tests; helpers plus tests; benchmark script plus these notes and JSON.

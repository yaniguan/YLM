# CPU smoke benchmarks

Tiny 3-seed, size-matched CPU runs of the two comparison scripts (`~638k–665k` parameters per model).
They use small built-in datasets, so treat them as smoke tests, not final results.

Run on CPU with `torch 2.14.0+cpu` and `e3nn 0.6.0`:

```bash
python scripts/35_build_and_eval_chiral_benchmark.py --tiny --epochs 24 --seeds 42 43 44 --match_params --out_dir <dir>
python scripts/32_benchmark_vs_unimol.py --tiny --epochs 20 --seeds 42 43 44 --match_params --out_dir <dir>
```

| File | Source |
| :--- | :--- |
| `chiral_benchmark.json`, `chiral_benchmark.md` | `benchmark_results.json` and `benchmark_table.md` from script 35 |
| `unimol_benchmark.json`, `unimol_benchmark.md` | `results.json` and `benchmark_table.md` from script 32 |

Numbers change slightly across PyTorch / `e3nn` versions and CPUs, so rerun the scripts before quoting them.

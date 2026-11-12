# CPU smoke benchmarks (Nov 12, 2026)

Tiny CPU runs of the two comparison scripts. They use small built-in datasets,
so treat them as smoke tests, not final results.

Run on CPU with Python 3.13, torch 2.15.0+cpu, e3nn 0.6.0:

```bash
python scripts/35_build_and_eval_chiral_benchmark.py --tiny --epochs 24 --out_dir <dir>
python scripts/32_benchmark_vs_unimol.py --tiny --epochs 20 --out_dir <dir>
```

| File | Source |
| :--- | :--- |
| `chiral_benchmark.json`, `chiral_benchmark.md` | `benchmark_results.json` and `benchmark_table.md` from script 35 |
| `unimol_benchmark.json`, `unimol_benchmark.md` | `results.json` and `benchmark_table.md` from script 32 |

Numbers change with the torch / e3nn version and CPU, so rerun the scripts before quoting them.

# Chiral (`0o`) + Multi-Rank (`0e + 1o + 2e`) Foundation Model Benchmark ($K=8$ $O(3)$ Rotations + Reflections)

Dataset splits (`ouroboros_enantiomers.extxyz` + `.manifest.jsonl`): `train=576`, `val=84`, `test=84`, `test_heldout=84`, `test_zero_shot_scaffold=112`.

## Table 1: Stereochemical Chirality (`0o` Pseudoscalar) & Symmetry Audit

| Model Family | Enantiomer Acc (Test) ↑ | Held-out Wording Acc ↑ | Zero-Shot Scaffold Acc ↑ | `0o` Parity Err ↓ | Achiral Planar `|f(X)|` ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **GraphMVP / 3D-InfoMax** |  52.1% |  47.9% |  50.0% | 6.92e-01 | 1.10e+00 |
| **Uni-Mol (3D Dist-Transformer)** |  52.1% |  52.1% |  48.4% | 7.62e-01 | 7.99e-01 |
| **EquiLLM (Inv-Bridge + 1o)** |  54.2% |  52.1% |  48.4% | 6.66e-01 | 7.35e-01 |
| **YLM-Universe FM (Ours)** | 100.0% | 100.0% | 100.0% | 4.27e-07 | 3.92e-08 |

## Table 2: Simultaneous Scalar (`0e`), Vector (`1o`), Tensor (`0e+2e`) & Routing Performance

| Model Family | Scalar MAE (`0e`) ↓ | Vec Cos (`1o`) ↑ | Vec $O(3)$ Eq Err ↓ | Tensor Cos (`0e+2e`) ↑ | Routing Acc ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **GraphMVP / 3D-InfoMax** | 0.0867 | +0.0743 | 1.32e+00 | +0.7201 | 0.833 |
| **Uni-Mol (3D Dist-Transformer)** | 0.0845 | +0.0488 | 1.33e+00 | +0.7218 | 0.833 |
| **EquiLLM (Inv-Bridge + 1o)** | 0.1261 | +0.6574 | 1.04e-06 | +0.9715 | 0.833 |
| **YLM-Universe FM (Ours)** | 0.0862 | +0.7275 | 9.07e-07 | +0.9982 | 0.854 |

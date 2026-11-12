# 3D Model Comparison ($K=8$ Random $O(3)$ Rotations + Mirror Flips)

## Table A: Invariant Scalar (`0e`) & Language-to-Atom Routing Performance

| Model Family | Params | Scalar MAE (`0e`) ↓ | Scalar Rot-Inv Err ↓ | Routing Acc (Test) ↑ | Routing Acc (Held-out) ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **GraphMVP / 3D-InfoMax** | 262,895 | 0.1678 | 0.00e+00 | 0.750 | 0.736 |
| **Uni-Mol (3D Dist-Transformer)** | 141,342 | 0.1895 | 2.59e-08 | 0.764 | 0.778 |
| **EquiLLM (Inv-Bridge + 1o)** | 123,573 | 0.1940 | 1.06e-08 | 0.653 | 0.722 |
| **YLM-Universe FM (Ours)** | 664,838 | 0.1754 | 4.85e-08 | 0.875 | 0.861 |

## Table B: Covariant Vector (`1o`: T1/T3/T4/T5) & Symmetric Tensor (`0e+2e`: T2) Performance

| Model Family | Vec Cos (`1o`) ↑ | Force Cos (T3) ↑ | Comp Diff/Proj Cos ↑ | Vec $O(3)$ Eq Err ↓ | Tensor Cos (`0e+2e`) ↑ | Held-out Vec Cos ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **GraphMVP / 3D-InfoMax** | +0.0351 | +0.0140 | +0.0681 | 1.31e+00 | +0.6500 | +0.0551 |
| **Uni-Mol (3D Dist-Transformer)** | +0.0661 | +0.0669 | +0.0747 | 1.31e+00 | +0.6586 | +0.0227 |
| **EquiLLM (Inv-Bridge + 1o)** | +0.5356 | +0.2554 | +0.6086 | 7.77e-07 | +0.9256 | +0.3095 |
| **YLM-Universe FM (Ours)** | +0.6292 | +0.4638 | +0.6233 | 8.05e-07 | +0.9915 | +0.4963 |

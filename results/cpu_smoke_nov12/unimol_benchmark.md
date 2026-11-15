# 3D Model Comparison ($K=8$ Random $O(3)$ Rotations + Mirror Flips)

Seeds: `42, 43, 44` (`mean ± std`); baseline widths: size-matched (`--match_params`).

## Table A: Invariant Scalar (`0e`) & Language-to-Atom Routing Performance

| Model Family | Params | Scalar MAE (`0e`) ↓ | Scalar Rot-Inv Err ↓ | Routing Acc (Test) ↑ | Routing Acc (Held-out) ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **GraphMVP / 3D-InfoMax** | 649,695 | 0.1823 ± 0.0096 | 0.00e+00 ± 0.00e+00 | 0.782 ± 0.056 | 0.778 ± 0.049 |
| **Uni-Mol (3D Dist-Transformer)** | 648,142 | 0.2125 ± 0.0046 | 1.28e-08 ± 8.19e-10 | 0.787 ± 0.024 | 0.796 ± 0.024 |
| **EquiLLM (Inv-Bridge + 1o)** | 638,213 | 0.1653 ± 0.0038 | 2.71e-08 ± 1.94e-09 | 0.755 ± 0.013 | 0.708 ± 0.020 |
| **YLM-Universe FM (Ours)** | 664,838 | 0.1506 ± 0.0096 | 3.94e-08 ± 1.22e-08 | 0.866 ± 0.040 | 0.782 ± 0.036 |

## Table B: Covariant Vector (`1o`: T1/T3/T4/T5) & Symmetric Tensor (`0e+2e`: T2) Performance

| Model Family | Params | Vec Cos (`1o`) ↑ | Force Cos (T3) ↑ | Comp Diff/Proj Cos ↑ | Vec $O(3)$ Eq Err ↓ | Tensor Cos (`0e+2e`) ↑ | Held-out Vec Cos ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **GraphMVP / 3D-InfoMax** | 649,695 | +0.0527 ± 0.0101 | +0.0416 ± 0.0095 | +0.0954 ± 0.0166 | 1.32e+00 ± 2.41e-02 | +0.6455 ± 0.0028 | +0.0417 ± 0.0086 |
| **Uni-Mol (3D Dist-Transformer)** | 648,142 | +0.0660 ± 0.0216 | +0.0460 ± 0.0356 | +0.1044 ± 0.0187 | 1.35e+00 ± 1.96e-02 | +0.6852 ± 0.0081 | +0.0501 ± 0.0101 |
| **EquiLLM (Inv-Bridge + 1o)** | 638,213 | +0.4459 ± 0.0399 | +0.2288 ± 0.0541 | +0.4114 ± 0.0494 | 8.00e-07 ± 5.64e-08 | +0.9457 ± 0.0119 | +0.3552 ± 0.0534 |
| **YLM-Universe FM (Ours)** | 664,838 | +0.5662 ± 0.0377 | +0.4634 ± 0.0490 | +0.4629 ± 0.0805 | 8.77e-07 ± 7.43e-08 | +0.9939 ± 0.0019 | +0.4769 ± 0.0251 |

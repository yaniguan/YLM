# Chiral (`0o`) + Multi-Rank (`0e + 1o + 2e`) Foundation Model Benchmark ($K=8$ $O(3)$ Rotations + Reflections)

Dataset splits (`ouroboros_enantiomers.extxyz` + `.manifest.jsonl`): `train=576`, `val=84`, `test=84`, `test_heldout=84`, `test_zero_shot_scaffold=112`. Seeds: `42, 43, 44` (`mean ± std`); baseline widths: size-matched (`--match_params`).

## Table 1: Stereochemical Chirality (`0o` Pseudoscalar) & Symmetry Audit

Models whose language readout receives only `0e` parity-even scalars (`GraphMVP`, `Uni-Mol`, `EquiLLM`) assign identical features to `(R)` and `(S)` mirror images and are marked `cannot represent` rather than scored.

| Model Family | Params | Enantiomer Acc (Test) ↑ | Held-out Wording Acc ↑ | Zero-Shot Scaffold Acc ↑ | `0o` Parity Err ↓ | Achiral Planar `|f(X)|` ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **GraphMVP / 3D-InfoMax** | 649,695 | cannot represent | cannot represent | cannot represent | cannot represent | 1.06e+00 ± 1.24e-01 |
| **Uni-Mol (3D Dist-Transformer)** | 648,142 | cannot represent | cannot represent | cannot represent | cannot represent | 3.08e-01 ± 1.41e-01 |
| **EquiLLM (Inv-Bridge + 1o)** | 638,213 | cannot represent | cannot represent | cannot represent | cannot represent | 4.40e-01 ± 2.82e-01 |
| **Uni-Mol + Signed-Volume (0o)** | 660,670 | 64.9 ± 26.1% | 64.6 ± 26.4% | 70.1 ± 21.3% | 1.05e-06 ± 5.93e-07 | 6.92e-07 ± 5.46e-07 |
| **YLM-Universe FM (Ours)** | 664,838 | 100.0 ± 0.0% | 100.0 ± 0.0% | 52.1 ± 26.2% | 4.65e-07 ± 8.27e-08 | 1.93e-07 ± 6.34e-08 |

## Table 2: Simultaneous Scalar (`0e`), Vector (`1o`), Tensor (`0e+2e`) & Routing Performance

| Model Family | Params | Scalar MAE (`0e`) ↓ | Vec Cos (`1o`) ↑ | Vec $O(3)$ Eq Err ↓ | Tensor Cos (`0e+2e`) ↑ | Routing Acc ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **GraphMVP / 3D-InfoMax** | 649,695 | 0.0906 ± 0.0085 | +0.0537 ± 0.0195 | 1.35e+00 ± 2.66e-02 | +0.7082 ± 0.0098 | 0.868 ± 0.020 |
| **Uni-Mol (3D Dist-Transformer)** | 648,142 | 0.2373 ± 0.0270 | +0.0474 ± 0.0305 | 1.29e+00 ± 6.37e-02 | +0.7625 ± 0.0013 | 0.882 ± 0.020 |
| **EquiLLM (Inv-Bridge + 1o)** | 638,213 | 0.0868 ± 0.0202 | +0.7069 ± 0.0149 | 1.16e-06 ± 1.28e-08 | +0.9789 ± 0.0036 | 0.861 ± 0.043 |
| **Uni-Mol + Signed-Volume (0o)** | 660,670 | 0.1487 ± 0.0293 | +0.0486 ± 0.0423 | 1.30e+00 ± 5.76e-02 | +0.7590 ± 0.0030 | 0.840 ± 0.010 |
| **YLM-Universe FM (Ours)** | 664,838 | 0.0981 ± 0.0095 | +0.7202 ± 0.0435 | 9.23e-07 ± 1.04e-07 | +0.9968 ± 0.0008 | 0.854 ± 0.017 |

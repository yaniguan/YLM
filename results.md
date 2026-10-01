# Results: TensorQA-v0

Mean ± std over seeds. 32 random rotations per test question (8 for the LLM baseline).
MAE units: dipole e·bohr, force hartree/bohr. eq_err = max over test questions of ||f(Rx) - R f(x)|| / ||f(x)||.

## Test (wordings seen in training)

| Model | data | seeds | dipole MAE | dipole cos | force MAE | force cos | atom acc | dipole MAE, orig. frame | eq_err |
|---|---|---|---|---|---|---|---|---|---|
| YLM (Option A) + MiniLM | 100% | 5 | 0.083 ± 0.003 | 0.980 ± 0.002 | 0.0075 ± 0.0006 | 0.946 ± 0.006 | 0.591 ± 0.336 | - | 7e-04 |
| YLM (Option A) + MiniLM | 30% | 1 | 0.091 | 0.975 | 0.0122 | 0.890 | 0.077 | - | 1e-04 |
| YLM (Option A) + MiniLM | 10% | 1 | 0.132 | 0.948 | 0.0184 | 0.799 | 0.698 | - | 2e-04 |
| YLM (Option A) + MiniLM | 3% | 1 | 0.203 | 0.897 | 0.0287 | 0.630 | 0.533 | - | 2e-04 |
| YLM (Option A) + MatSciBERT | 100% | 5 | 0.082 ± 0.002 | 0.980 ± 0.001 | 0.0064 ± 0.0003 | 0.960 ± 0.004 | 0.941 ± 0.023 | - | 1e-03 |
| YLM (Option A) + MatSciBERT | 30% | 1 | 0.092 | 0.972 | 0.0072 | 0.951 | 0.993 | - | 5e-04 |
| YLM (Option A) + MatSciBERT | 10% | 1 | 0.134 | 0.948 | 0.0249 | 0.713 | 0.237 | - | 2e-04 |
| YLM (Option A) + MatSciBERT | 3% | 1 | 0.198 | 0.908 | 0.0410 | 0.422 | 0.334 | - | 1e-04 |
| Baseline v: task ID instead of text | 100% | 4 | 0.067 ± 0.003 | 0.986 ± 0.001 | 0.0424 ± 0.0000 | 0.113 ± 0.004 | 0.042 ± 0.060 | - | 2e-04 |
| Baseline v: task ID instead of text | 30% | 1 | 0.097 | 0.972 | 0.0426 | 0.092 | 0.129 | - | 6e-05 |
| Baseline v: task ID instead of text | 10% | 1 | 0.132 | 0.953 | 0.0431 | 0.062 | 0.109 | - | 2e-04 |
| Baseline v: task ID instead of text | 3% | 1 | 0.187 | 0.913 | 0.0438 | 0.039 | 0.042 | - | 3e-04 |

## Test, held-out wordings

| Model | data | seeds | dipole MAE | dipole cos | force MAE | force cos | atom acc | dipole MAE, orig. frame | eq_err |
|---|---|---|---|---|---|---|---|---|---|
| YLM (Option A) + MiniLM | 100% | 5 | 0.301 ± 0.059 | 0.726 ± 0.140 | 0.0136 ± 0.0021 | 0.859 ± 0.040 | 0.533 ± 0.303 | - | 5e-04 |
| YLM (Option A) + MiniLM | 30% | 1 | 0.420 | 0.548 | 0.0143 | 0.858 | 0.072 | - | 5e-04 |
| YLM (Option A) + MiniLM | 10% | 1 | 0.292 | 0.819 | 0.0212 | 0.753 | 0.661 | - | 1e-04 |
| YLM (Option A) + MiniLM | 3% | 1 | 0.287 | 0.849 | 0.0299 | 0.593 | 0.522 | - | 1e-04 |
| YLM (Option A) + MatSciBERT | 100% | 5 | 0.179 ± 0.048 | 0.941 ± 0.032 | 0.0092 ± 0.0012 | 0.917 ± 0.021 | 0.911 ± 0.008 | - | 8e-04 |
| YLM (Option A) + MatSciBERT | 30% | 1 | 0.167 | 0.962 | 0.0135 | 0.854 | 0.880 | - | 2e-04 |
| YLM (Option A) + MatSciBERT | 10% | 1 | 0.282 | 0.827 | 0.0260 | 0.679 | 0.241 | - | 1e-04 |
| YLM (Option A) + MatSciBERT | 3% | 1 | 0.395 | 0.665 | 0.0395 | 0.402 | 0.333 | - | 2e-04 |
| Baseline v: task ID instead of text | 100% | 4 | 0.067 ± 0.003 | 0.986 ± 0.001 | 0.0424 ± 0.0000 | 0.113 ± 0.004 | 0.042 ± 0.060 | - | 2e-04 |
| Baseline v: task ID instead of text | 30% | 1 | 0.097 | 0.972 | 0.0426 | 0.092 | 0.129 | - | 5e-05 |
| Baseline v: task ID instead of text | 10% | 1 | 0.132 | 0.953 | 0.0431 | 0.062 | 0.109 | - | 2e-04 |
| Baseline v: task ID instead of text | 3% | 1 | 0.187 | 0.913 | 0.0438 | 0.039 | 0.042 | - | 3e-04 |


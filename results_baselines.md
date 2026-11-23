# Post-v0 baselines (not part of the v0.1.0 release)

Mean ± std over seeds (n = number of runs). MAE: force in hartree/bohr, dipole in e·bohr.

## v0 questions (test)

| Model | n | force MAE | force cos | dipole cos | rotation error (mean) |
|---|---|---|---|---|---|
| Pipeline: rule parser -> atom index -> equivariant force model | 3 | 0.0057 ± 0.0001 | 0.970 | 0.978 | 2.6e-06 |
| Scalar collapse (invariants only, rotation-augmented) | 3 | 0.0426 ± 0.0000 | -0.000 | 0.024 | 1.3e+00 |
| Qwen2.5-0.5B coordinates as text, no hint, no aug (v1) | 1 | 0.0425 | 0.005 | 0.461 | 1.7e+00 |
| Qwen2.5-0.5B coordinates as text, no hint, aug (v1) | 1 | 0.0425 | -0.000 | 0.052 | 1.7e+00 |
| Qwen2.5-0.5B, no hint, lr 5e-5, 5 epochs (v2) | 1 | 0.0425 | 0.004 | 0.105 | 1.9e+00 |
| Qwen2.5-0.5B, atom hint, no aug (v2) | 1 | 0.0425 | -0.000 | 0.015 | 1.4e+00 |
| Qwen2.5-0.5B, atom hint, aug (v2) | 1 | 0.0425 | -0.003 | -0.019 | 1.4e+00 |
| Qwen2.5-0.5B, atom hint, lr 2e-5 / head 1e-4 (v3) | 0 | not run (queued multi-GPU; see Issues #56, #58) | - | - | - |
| Qwen2.5-0.5B, no hint, lr 2e-5 / head 1e-4 (v3) | 0 | not run (queued multi-GPU; see Issues #56, #58) | - | - | - |

Parser accuracy (force questions): {'val': 1.0, 'test': 1.0, 'test_heldout': 1.0}


## Compositional questions (T4 diff = F_A - F_B, T5 proj = signed projection of F_A along A->B)

| Model | split | n | T4 cos | T4 MAE | T5 cos | T5 MAE |
|---|---|---|---|---|---|---|
| Option B/C-lite | test | 3 | 0.8897 ± 0.0918 | 0.0190 ± 0.0096 | 0.7808 ± 0.1064 | 0.0087 ± 0.0047 |
| Option B/C-lite | test_heldout | 3 | 0.0451 ± 0.0082 | 0.0672 ± 0.0028 | 0.5743 ± 0.3741 | 0.0148 ± 0.0116 |
| Option A | test | 3 | 0.7586 ± 0.0057 | 0.0344 ± 0.0005 | 0.5597 ± 0.0040 | 0.0196 ± 0.0002 |
| Option A | test_heldout | 3 | -0.3835 ± 0.2445 | 0.1094 ± 0.0124 | 0.1629 ± 0.2369 | 0.0303 ± 0.0066 |

## Language-geometry interface generalization (`scripts/64_interface_generalization_study.py`, 3 seeds)

All four interfaces share the same frozen 60-epoch QM9 `Encoder` (`checkpoints/v0/ylm_matscibert_s0.pt`) and frozen `MiniLM-L6-v2` token embeddings on molecule-disjoint official `TensorQA-v0.1.0` GFN2-xTB splits:

| Interface | `rigid_single` cos | `ood_synonym_single` cos | `ood_synonym_comp` cos | `group_ood` cos | Rule parse rate (ID / OOD) |
|---|---|---|---|---|---|
| Rule parser + O(3) executor (`rule_exec`) | 0.8856 ± 0.0053 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.980 / 0.000 |
| Option A continuous (`optA_cont`) | 0.3687 ± 0.0063 | 0.3429 ± 0.0254 | 0.1474 ± 0.0176 | 0.7954 ± 0.0162 | learned |
| Option B/C-lite continuous (`optBC_cont`) | 0.4028 ± 0.1497 | 0.3730 ± 0.1559 | 0.1838 ± 0.0926 | 0.7474 ± 0.0631 | learned |
| Neuro-symbolic + O(3) executor (`GroupProgModel`) | 0.8718 ± 0.0107 | 0.7984 ± 0.0775 | 0.5162 ± 0.0511 | 0.8687 ± 0.0321 | learned |

## Cross-attention parameter & system-size scaling (`scripts/66_scaling_benchmark.py`)

| Architecture | Cross-attn params | Checkpoint (fp32) | `N=15` ms | `N=64` ms | `N=180` ms | `N=400` ms |
|---|---:|---:|---:|---:|---:|---:|
| Option A | 85,600 | 1.30 MB | 20.64 | 78.32 | 314.12 | 652.06 |
| Option C (channelwise) | 684,960 | 3.59 MB | 28.93 | 84.75 | 280.15 | 688.71 |
| Option C (fully_connected) | 12,881,568 | 50.12 MB | 39.71 | 97.89 | 273.40 | 603.42 |

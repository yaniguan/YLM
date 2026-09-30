# Progress log

Newest first. One entry per work session.

## 2026-09-30 (weeks 1-3 of the plan)

Done:
- Novelty re-check: read EquiLLM, EquiVLA, CatalyticMLLM, MatterChat. All four keep language on invariants
  only. Claim still holds. Details in `02_novelty_check.md`.
- Target venue changed to ICML 2027 (ICLR 2027 deadline of 2026-09-25 had passed).
- Labels: 20k QM9 molecules, displaced by 0.05 A noise, GFN2-xTB -> dipole vector + forces (13 s on the Mac).
- TensorQA-v0: 84,757 questions (dipole + "force on <described atom>"), 584 unique strings,
  held-out templates for `test_heldout`.
- Model: equivariant encoder (NequIP-style, l<=2, with 0o/1e) + Option A cross-attention + l=1 readout.
  342k parameters. Equivariance test passes (error < 1e-9 in float64).
- Text caches: MiniLM (384-d) and MatSciBERT (768-d), frozen.
- Baseline (ii) script written: Qwen2.5 reads coordinates as text, regression head, with/without rotation augmentation.

Findings worth remembering:
- GFN2-xTB polarizability is bad (water: 2.4 vs ~9.8 bohr^3). Polarizability labels need DFT (PySCF) in week 5.
- DFT test (PySCF, B3LYP/def2-SVPD, finite field): water alpha 9.6 bohr^3 (exp ~9.8), good.
  Cost 2-14 min per QM9 molecule on the Mac -> ~5-7k CPU-hours for 20k. Options: analytic CPHF (~3x cheaper),
  5k-molecule subset, Hoffman2 CPU array job. Decide at the next check-in.
- QM9 geometries are relaxed, so forces there are ~0. We displace atoms to get real forces.
- e3nn `FullyConnectedTensorProduct` with per-edge weights was ~40x too slow. Switched to channel-wise ("uvu") product + Linear.
- Hoffman2: this account can only use V100 / RTX2080Ti (A100/H100/L40S/H200 refused; gpu_h100.q disabled today).
  CentOS 7 (glibc 2.17): env uses pip torch 2.5.1+cu124, `--only-binary=:all:`. Login node kills big conda/pip jobs (memory).
- Expanse: SSH key not installed yet (needs `ssh-copy-id yguan1@login.expanse.sdsc.edu` once, by hand).

Running:
- Hoffman2: 15 jobs queued (MiniLM / MatSciBERT / task-ID x 3 seeds; 10% and 30% data). GPU pool was full.
- Mac (MPS): `local_minilm_s0`, `local_taskid_s0` (~3.5 h each).

Next:
- Month-1 checkpoint table: ours vs task-ID vs coordinates-as-text, on dipole and force, with rotation protocol.

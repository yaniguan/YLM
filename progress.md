# Progress log

Newest first. One entry per work session.

## 2026-10-11 (v0.1.0 + post-v0 baselines)

Done:
- Released v0.1.0 (tag + GitHub release): 3 checkpoints, `scripts/09` (re-evaluating the MatSciBERT checkpoint
  reproduces training numbers: force MAE 0.00684 vs 0.00686; rotated-molecule demo diff ~2e-7).
- Read Mol-StrucTok (KDD '26): invariant spherical-coordinate tokens for generation, not irreps-to-text. Cited, claim unchanged.
- Code added (smoke-tested on CPU, tests pass, **no result numbers yet**): pipeline and scalar-collapse baselines
  (`src/baselines.py`, `04_train.py --model`), rule parser (100% on v0 test and held-out force questions),
  compositional questions + Option B/C-lite, quadrupole labels.
- Hoffman2 queue (all V100): Qwen aug / noaug (15130177/8), pipe_s0-2 + scalar_s0-2 (15130204-9), comp_bc/optA_s0-2 (15130255-60).
  Old m1_* jobs from 09-30 are still `qw` there and were not touched (results already exist from Expanse).

Notes / corrections:
- The original QM7b file has scalar polarizabilities only. Full tensors are in Wilkins et al. 2019 (Materials Cloud,
  availability still to verify); QM7-X also lists polarizability tensors. Quadrupole (PySCF single point) is the cheap l=2 fallback.
- The v0 questions are fully parseable by regex, so the pipeline baseline will likely match YLM on v0. That is why T4/T5 exist.
- The scalar-collapse baseline is a proxy for invariant-token methods, not a reproduction of Mol-StrucTok.
- Expanse SSH key is still not installed (needs `ssh-copy-id` by hand), so nothing was submitted there.

## 2026-10-01 (v0 results)

Month-1 checkpoint: **passed.**
- 23 runs on Expanse V100 (~35 min each): YLM + MiniLM / MatSciBERT (5 seeds), task ID (4 seeds), 3/10/30% data.
- Force on a described atom: YLM + MatSciBERT MAE 0.0064, cos 0.96; task ID 0.0424, cos 0.11.
  The text is what makes atom selection possible.
- Rotation error mean ~2e-6 for all models (float32), no augmentation.
- MatSciBERT > MiniLM on held-out wordings (dipole cos 0.94 vs 0.73). MiniLM better with <=10% data.
- Attention peaks on the asked atom or a bonded neighbour 94-97% of the time; for H atoms it looks at the
  heavy atom the H is bonded to. Strict "exact atom" accuracy is noisy across seeds (0.08-0.94), so it is not
  a good headline metric.
- Task ID is slightly better on the dipole (0.067 vs 0.082): language has a small cost when no selection is needed.

Fixed along the way:
- Expanse Slurm needs `--nodes=1 --ntasks-per-node=1`; max ~20 queued jobs per account (shared with other work).
- `evaluate()` device bug on CUDA (scale tensor on CPU).
- Mac MPS runs out of memory when other apps are open; Mac is not used for training any more.
- LLM baseline (Qwen 0.5B, float32, V100): batch 32 runs out of memory. Now batch 8 + gradient checkpointing.

Open:
- Coordinates-as-text LLM baseline (with / without rotation augmentation): re-running.
- Next (after v0): T4/T5 compositional questions, Option B/C, l=2 targets (quadrupole via PySCF, or Wilkins et al. polarizability tensors).

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

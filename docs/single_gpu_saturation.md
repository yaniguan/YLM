# Single-GPU saturation (one H200 or B200)

Status: plan. Nothing here has been measured yet. All targets are TARGETS; all sizes are rough estimates.

Related: [lab_scale_plan.md](lab_scale_plan.md) (what we train), [lab_cost_setup.md](lab_cost_setup.md) (what it costs).

## Goal

Keep ONE big GPU busy all the time during post-training (SFT -> DPO -> GRPO) and science workloads. An idle rented GPU is wasted money.

| GPU | memory | HBM BW | BF16 dense | FP8 dense | FP4 dense | TDP | source |
|---|---|---|---|---|---|---|---|
| H200 SXM | 141 GB HBM3e | 4.8 TB/s | 989-990 TFLOPS | 1,979 TFLOPS | n/a (no FP4 tensor cores on Hopper) | up to 700 W | nvidia.com/en-us/data-center/h200, lenovopress.lenovo.com/lp1944 |
| B200 (HGX, SXM6) | 180 GB HBM3e usable | 7.7 TB/s | 2,250 TFLOPS | 4,500 TFLOPS | 9,000 TFLOPS | up to 1,000 W | NVIDIA Blackwell datasheet, lenovopress.lenovo.com/lp2226 |

- Dense = no sparsity. NVIDIA often lists sparse numbers; dense is half.
- B200: early marketing cites 192 GB physical. HGX B200 exposes 180 GB. We size for 180 GB.
- Rent (Nov 2026, estimates): H200 $3.59/hr (RunPod Community), $5.29 (RunPod Secure); B200 $5.98 (RunPod Community), $7.69-7.99 (Lambda / RunPod Secure). Sources: runpod.io/pricing, joulemath.com/gpu, gpuwerk.com/blog/h100-h200-b200-rental-prices.

## 1. Software and science architecture

Idea: put every stage on the same GPU and keep it in a loop, so the GPU never waits for another machine.

**GRPO, colocated mode**
- Rollouts run in an inference engine (vLLM or SGLang) on the same GPU as the trainer.
- No second model copy kept live: the engine sleeps (frees weights + KV cache) while the trainer runs, then wakes.
- After each optimizer step, sync new weights into the engine (in-GPU copy, no disk).
- With LoRA, the reference model is the base with the adapter turned off (no extra copy).
- Then train on the same GPU. Recipe stays SFT -> DPO -> GRPO.
- GRPO code and protocol envs: [labbench-posttrain](https://github.com/yaniguan/labbench-posttrain) (`posttrain/grpo.py`, `posttrain/rewards.py`, `bioenvs/`). Reuse, do not copy.

**Science calls as batched rewards / verifiers (same loop)**

| reward / verifier | runs on | batching rule |
|---|---|---|
| [WignerFlow](https://github.com/yaniguan/WignerFlow) surrogate rollouts (argon only today) | GPU | many short rollouts in one batch per GRPO group |
| [ouroboros-ocsr](https://github.com/yaniguan/ouroboros-ocsr) inference (image -> SMILES) | GPU | one batched call for all images in the step |
| RDKit checks (valid SMILES, stereo, formula) | CPU worker pool | run on finished rollout chunks while GPU decodes |
| labbench-posttrain protocol rewards (QA, ordering, error fix) | CPU | same as RDKit |

**Feeding the GPU**
- Sequence packing: pack short samples into full-length rows; no padding waste.
- Largest micro-batch that fits; then grad accumulation to reach the global batch.
- Async data loading: `pin_memory=True`, `num_workers > 0`, prefetch the next batch.
- Tokenize and pack on CPU ahead of time, not inside the step.

**One GRPO step (ASCII timeline, time ->)**

```
GPU main  |wake+sync W|== rollout decode (CUDA graphs) ==|== sci rewards: WF + OCSR ==|sleep|== policy fwd/bwd (packed) ==|opt|
CPU pool  |           |   [RDKit + protocol checks on finished chunks ......]          |     [tokenize + pack batch k+1 ......]
H2D side  |           |                                                                |     [copy batch k+1, non_blocking ]
```

- Overlap 1: CPU rewards start on finished rollout chunks; the GPU keeps decoding.
- Overlap 2: next batch is packed and copied on a side stream during fwd/bwd.
- Gaps to kill: wake/sleep, weight sync, waiting for the slowest rollout (cap length; chunk the group).

## Memory sizing (rough estimates, not measured)

Rules of thumb (bytes per parameter, before activations):

| method | what is stored | bytes / param |
|---|---|---|
| Full FT, AdamW, mixed precision | BF16 weights 2 + BF16 grads 2 + FP32 master 4 + 2 Adam states 8 | ~16 |
| LoRA | frozen BF16 weights 2 + small adapter (~1% params x 16) | ~2.2 |
| QLoRA (4-bit base) | 4-bit weights 0.5-0.6 + small adapter | ~0.7 |

Reserve: we keep ~40 GB free for activations (training phase) or rollout KV cache (rollout phase). With sleep mode, the two phases do not hold memory at the same time. 40 GB is an assumption.

| method | 141 GB (H200) | 180 GB (B200) |
|---|---|---|
| Full FT | (141 - 40) / 16 = ~6B | (180 - 40) / 16 = ~8-9B |
| LoRA | (141 - 40) / 2.2 = ~45B | (180 - 40) / 2.2 = ~63B |
| QLoRA | (141 - 40) / 0.7 = ~140B | (180 - 40) / 0.7 = ~200B |

- These are upper bounds. Long context, big micro-batch or big groups need more reserve.
- Bigger models need activation checkpointing to fit the 40 GB reserve.
- LoRA/QLoRA + GRPO: the rollout engine still needs the base weights at inference precision when awake.
- QLoRA rollouts are slow; use QLoRA mainly for SFT/DPO of big models.
- Our planned 1-8B student (see [lab_scale_plan.md](lab_scale_plan.md)) fits full FT on one B200; on one H200 the top of that range likely needs LoRA or more checkpointing.
- Check real peaks with `scripts/profile_gpu.py` before planning a run.

## 2. GPU hardware optimization

**Precision**
- BF16 is the default for training and rollouts.
- FP8 via Transformer Engine on Hopper and Blackwell (training matmuls; check loss curves vs BF16).
- FP4 (NVFP4) on Blackwell only. Use it for inference/rollouts first, training later if at all.
- Report MFU against the peak of the precision you actually run.

**Kernels**
- FlashAttention-3 on Hopper; use the Blackwell FlashAttention build where supported.
- Liger kernel: fused RMSNorm, RoPE, SwiGLU, fused linear cross-entropy (no full logits tensor in memory).
- CUDA graphs for decode and other small fixed-shape steps (cuts launch overhead).
- Activation checkpointing: saves memory, costs about one extra forward pass. Turn on only if it buys a bigger batch.

**Data movement**
- Pinned host memory + `non_blocking=True` copies.
- Copies on a separate CUDA stream so H2D overlaps compute.
- Keep data on the GPU between stages (rollout -> reward -> train); avoid `.cpu()` in the loop.

**Equivariant GNN / flow parts (YLM irreps attention, WignerFlow)**
- Fused tensor-product kernels instead of many small e3nn ops.
- Bigger graph batches: pack many small molecules per batch; pad-free scatter by graph index.
- Precompute spherical harmonics and Clebsch-Gordan coefficients when geometry is fixed (YLM computes SH each forward in [src/encoder.py](../src/encoder.py)).
- Keep e3nn paths in FP32 where accuracy needs it (equivariance error, forces); BF16 elsewhere.

**Measure, do not guess**
- NVTX ranges around stages (rollout, reward, fwd/bwd, opt, sync).
- Nsight Systems (`nsys`) for the timeline and idle gaps.
- Nsight Compute (`ncu`) for single hot kernels.
- `torch.profiler` for op-level tables (this repo already parses them: [scripts/analyze_profiling.py](../scripts/analyze_profiling.py)).
- MFU (dense transformer training) = 6 x N_params x tokens/s / peak dense FLOPS.
- HBM bandwidth utilization = bytes moved / s / peak HBM BW. Decode is usually bandwidth-bound, so watch this, not MFU.
- NVML "GPU util" only says a kernel was running. 100% util can still be low MFU. Track both.

## 3. Target metrics (all TARGET, not measured)

| metric | how measured | TARGET |
|---|---|---|
| GPU util (NVML), steady state | `profile_gpu.py` / `nvidia-smi dmon` | > 90% |
| MFU, LLM training (BF16) | 6·N·tokens/s / 989 (H200) or 2,250 (B200) TFLOPS | 35-50% |
| MFU, equivariant GNN / flow | same idea, FLOPs counted per model | TBD after first profile |
| Peak memory | `torch.cuda.max_memory_allocated` / HBM | 80-90% of HBM |
| HBM BW util, rollout decode | bytes moved / s / 4.8 or 7.7 TB/s | > 60% |
| Rollout tokens/s | engine logs | TBD baseline; then >= 2x the untuned baseline |
| Idle gaps in nsys timeline | gap time / step time | < 5% |
| Wake + weight sync per GRPO step | NVTX range | < 10% of step time |

No result rows yet. Fill a results table only from real `profile_gpu.py` / `nsys` runs, with GPU, date and git hash.

## How to measure

```bash
# CI / laptop: plan only, or a tiny real run on CPU (MFU and NVML are null on CPU)
python scripts/profile_gpu.py --dry-run
python scripts/profile_gpu.py --device cpu --steps 3 --tokens 256 --d-model 128

# On the rented GPU: built-in toy step, BF16 peak, JSONL out (runs/ is git-ignored)
python scripts/profile_gpu.py --steps 50 --nvtx --out runs/profile_toy.jsonl

# Your own workload: factory(torch, device, dtype) -> {step, items_per_step, flops_per_step?, bytes_per_step?}
python scripts/profile_gpu.py --workload mypkg.grpo_step:make --precision fp8 --out runs/grpo.jsonl

# Timeline with NVTX ranges
nsys profile -t cuda,nvtx -o runs/grpo_step python scripts/profile_gpu.py --nvtx --steps 10
```

- Logs per step: seconds, items/s, achieved TFLOPS, MFU, HBM BW util, peak memory, NVML GPU/memory util.
- Peak table: H200 and B200 only (from the spec table above). Other GPUs: `--peak-tflops`.
- `pip install nvidia-ml-py` for NVML fields; otherwise the log says "nvml unavailable".
- Do not commit profile outputs.
- Tests: `python -m pytest -q tests/test_profile_gpu.py`.

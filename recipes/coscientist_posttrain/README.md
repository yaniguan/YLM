# Co-scientist post-training on one GPU

A recipe for post-training a lab co-scientist model on a single H200 or B200:
data -> SFT -> DPO (optional) -> GRPO with sealed verifiers -> OPSD.
It shows how text, 3D geometry, images and simulation results enter the model,
how to keep the GPU busy, and how to profile it.

**Status.** Only the CPU smoke path has been run. No GPU number has been measured.
Every GPU number below is a TARGET or an estimate from spec sheets.

Tags used everywhere (records and configs):

| tag | meaning |
|---|---|
| MEASURED | came from a real run; must name a source file |
| TARGET | what we aim for; not measured |
| EXAMPLE | toy value for the schema or the CPU smoke test |

## Files

| file | what it does |
|---|---|
| `schema.py` | record schema v1: placeholders, units (angstrom -> bohr), license fields, argon-only sims, tag rules |
| `collate.py` | render chat turns, expand placeholders, pack samples into rows, build labels |
| `geom_tokens.py` | geometry -> k soft tokens (YLM encoder + `IrrepsToText` pooler + projector) |
| `verifiers.py` | sealed verifiers with timeouts; OPSD spec-leak guard |
| `workloads.py` | step factories for profiling (`sft_step`, `prefill_step`, `decode_step`, `geom_sft_step`) |
| `profile.py` | timed steps + torch.profiler, chrome trace, JSONL summary with MFU/MBU |
| `launch.sh` | `smoke`, `sft`, `dpo`, `grpo`, `opsd`, `profile` |
| `configs/*.json` | `cpu_smoke`, `colab`, `h200`, `b200`, `stages` |
| `examples/records.jsonl` | 6 example records |
| `tests/test_coscientist_recipe.py` | CPU smoke test (runs in CI) |

## Decisions

- **Model size.** Default: a <= 4B decoder, full fine-tune (`full_ft_4b` in
  `configs/h200.json` and `configs/b200.json`). Alternative: 7-8B with LoRA
  (`lora_8b` in the same files). Colab stays on `lora_small`.
- **DPO.** Optional. It stays in the pipeline only if verifier pass@1 goes up by
  at least 2 points over SFT; otherwise GRPO starts from the SFT checkpoint.

## 1. Pipeline and stage gates

```
data: token factory (#80: generate -> verify -> keep) + protocol QA (#78) + seeded-error twins + SQL + Lean tasks
  -> SFT (assistant tokens only, packed)
  -> DPO (optional)
  -> GRPO / RLVR (rewards only from sealed verifiers)
  -> OPSD (on-policy self-distillation)
```

- **Seeded-error twins.** Take a verified protocol. Inject one known error
  (wrong temperature, time, unit or amount; swapped or missing step).
  The error type and position are known, so "find and fix the error" gets an exact reward.
  The pair (correct, twin) is also a free DPO pair.
- **Sealed verifiers.** The model never sees verifier code, hidden tests or the `spec`.
  Each check runs in its own process with a timeout.
  `verify(record, completion) -> {"reward", "passed", "kind", "detail"}`.
  - protocol: seeded twins and step order here; full checks call the labbench-posttrain `bioenvs/` envs.
  - lean: typecheck with the Lean 4 protocol DSL in `lean/` (pass = no errors, no `sorry`).
  - sql: run on an in-memory sqlite fixture in read-only mode and compare result sets.
  - exact / numeric: normalized match or tolerance.
- **Why verifiers, not a learned judge.** In our earlier runs DPO preference accuracy was 90-97%
  while exact match stayed at 0-2%. A model can learn to prefer the right answer and still not produce it,
  so rewards must be checkable.
- **OPSD.** Same-model self-distillation only teaches something if the teacher sees more than the student.
  Teacher context = prompt + reference answer + verifier pass/fail and detail. Never the verifier `spec`.
  Student data = prompt + teacher answer only. `opsd_teacher_messages` and `opsd_student_record`
  raise `SpecLeak` if spec text gets in.
- Loss code for SFT, DPO, GRPO and OPSD lives in [labbench-posttrain](https://github.com/yaniguan/labbench-posttrain)
  (`posttrain/`). This recipe does not copy it.

Stage gates (all TARGET):

| gate | must pass before the next stage |
|---|---|
| data | every kept record passes its verifier; dedup; no 13-gram overlap with eval; `license` on every record |
| SFT | format-valid >= 95%; verifier pass@1 above the base model; no drop > 2 points on a general held-out set |
| DPO (optional) | verifier pass@1 up >= 2 points over SFT; else skip DPO and start GRPO from SFT |
| GRPO | pass@1 up; pass@8 not below SFT (no collapse); KL to reference bounded; hidden-test audit for reward hacking; answer length not exploding |
| OPSD | keeps >= 90% of the GRPO pass@1 gain, with fewer tokens per answer or a smaller student |

## 2. Data and licenses

- BioProBench is **CC-BY-NC 4.0: research use only**, not commercial.
- Every record has `source.license` and `source.commercial_ok`.
  `load_jsonl(path, commercial=True)` drops every record with `commercial_ok: false`.
- Commercial-safe path: our own protocols, seeded twins of them, and self-generated records that pass a verifier.
  The generator model's license must also allow training on its outputs.

## 3. Multimodal interface

One JSONL record per sample (`schema: coscientist.v1`). Placeholders in message text point into arrays:

| placeholder | array | how it enters the model |
|---|---|---|
| plain text | `messages` | protocols, papers, lab notes as chat turns |
| `<geom:i>` | `geoms` | `{Z, pos, unit}` -> YLM E(3) encoder -> k soft tokens |
| `<img:i>` | `images` | OCSR result inline as text: `[image 0: SMILES=O; ocsr_flag=0.04]` |
| `<sim:i>` | `sims` | WignerFlow summary numbers inline as text: `[sim 0: Ar, T_K=120]` |

Rules checked by `schema.py`:
- Placeholder indices must be exactly `0..len-1` of their array.
- Geometry is stored in bohr (the YLM encoder unit). Angstrom input is converted. Silent angstrom input would make the cutoff 1.9x too large.
- `sims` must be WignerFlow with `system: "Ar"`. Public WignerFlow results are argon only.
- Images whose VERDICT flag is above `max_ocsr_flag` (0.5, TARGET) are dropped from training.

**Geometry -> soft tokens** (`geom_tokens.py`):
```
(Z, pos) -> Encoder (per-atom irreps) -> IrrepsToText with k learned queries (attention pooler)
         -> invariants (scalars + norms of l>=1 channels) -> MLP projector -> (G, k, d_llm)
```
`<geom:i>` expands to k `<|geom|>` slots. The soft tokens are written into the input embeddings at those slots.
Only invariants enter the LLM, so the tokens do not change under rotation, reflection, translation or atom permutation
(checked in the smoke test). The equivariant pooled features are also returned, for an optional vector readout.
Soft tokens work as prompt embeddings in an inference engine, so the same path serves GRPO rollouts.
The YLM cross-attention path (text queries atoms inside the LLM) does not run inside vLLM/SGLang today,
so it stays for SFT and eval only.

Images: v1 passes OCSR output as text. A vision embedding can use the same slot trick later.

**Collator** (`collate.py`): render -> expand placeholders -> tokenize -> first-fit-decreasing packing into rows of `max_tokens`.
- `position_ids` restart at 0 for every sample.
- `cu_seqlens` + `max_seqlen` go to varlen FlashAttention, so samples in one row never attend to each other.
  On CPU, `attention_bias(cu_seqlens)` builds the same block-diagonal causal mask.
- `labels` are -100 on system/user turns, on `<|geom|>` slots, and on the first token of every sample
  (so the last token of sample i never learns to predict sample i+1).
- `geom` = all geometries in the row, padded (`src.ylm_lm.collate_geometry`); `geom_slots (G, k)` = where to write them.
- The smoke test uses a byte tokenizer (no download). A real tokenizer needs `encode`, `decode`, `special`, `vocab_size`.

Example records (shortened; full lines in `examples/records.jsonl`):
```json
{"id":"tw-017","task":"error_fix",
 "messages":[{"role":"user","content":"Find the error: 1. Thaw on ice. 2. Incubate at 95 C for 1 h. 3. Store at -20 C."},
             {"role":"assistant","content":"Step 2: 95 C denatures the enzyme; use 37 C."}],
 "verifier":{"kind":"protocol","spec":{"error_step":2,"field":"temperature","correct":"37 C"}},
 "pair":{"chosen":"Step 2 ... use 37 C.","rejected":"No error."},
 "source":{"origin":"seeded_twin","license":"own","commercial_ok":true}}
{"id":"gq-003","task":"geom_qa",
 "messages":[{"role":"user","content":"<geom:0> Is this molecule chiral? Image: <img:0>"},{"role":"assistant","content":"No."}],
 "geoms":[{"Z":[8,1,1],"pos":[[0,0,0],[1.81,0,0],[-0.45,1.75,0]],"unit":"bohr"}],
 "images":[{"uri":"crops/p12_f3.png","smiles":"O","ocsr_flag":0.04}],
 "verifier":{"kind":"exact","spec":{"answer":"No"}}}
{"id":"sim-009","task":"sim_qa",
 "messages":[{"role":"user","content":"<sim:0> Is the fluid above its boiling point at 1 atm?"},{"role":"assistant","content":"Yes."}],
 "sims":[{"engine":"wignerflow","system":"Ar","values":{"T_K":120.0},"tag":"EXAMPLE"}]}
```
`sim-009` is a schema placeholder (argon boils at 87.3 K at 1 atm), not a WignerFlow result.

**Inference API** (shape only; not built yet):
```
POST /v1/coscientist
{"messages": [...], "geoms": [...], "images": [...], "sims": [...],
 "max_tokens": 512, "verify": ["protocol", "sql"]}
->
{"text": "...",
 "verifications": [{"kind": "protocol", "passed": true, "reward": 1.0, "detail": "..."}],
 "flags": {"ocsr": [0.04], "sim_scope_ok": true},
 "usage": {"prompt_tokens": 0, "soft_tokens": 8, "completion_tokens": 0},
 "model": {"id": "...", "stage": "grpo", "commercial_ok": true}}
```
The server validates the request with `schema.validate`, builds soft tokens, calls the engine with prompt embeddings,
then runs the requested verifiers on the answer. It refuses `sims` outside argon.

## 4. Keeping one GPU busy (roofline view)

The key fact: math is cheap, moving bytes from HBM is expensive.
For any step, count FLOPs and bytes read. Their ratio is the arithmetic intensity.
The **ridge point** = peak FLOP/s / HBM bytes/s. Below it the step is memory-bound; above it, compute-bound.

| GPU | BF16 dense | FP8 dense | FP4 dense | HBM | ridge BF16 | ridge FP8 | ridge FP4 |
|---|---|---|---|---|---|---|---|
| H100 SXM (reference) | 989 TFLOPS | - | - | 3.35 TB/s | ~295 FLOP/byte | - | - |
| H200 SXM | 989 | 1979 | - | 4.8 TB/s | ~206 | ~412 | - |
| B200 HGX | 2250 | 4500 | 9000 | 7.7 TB/s | ~292 | ~584 | ~1169 |

Peaks are vendor dense numbers (same table as `docs/single_gpu_saturation.md` in PR #79).

What this means for each phase:
- **Decode** (rollouts) reads every weight once per step and does 2 FLOPs per weight per sequence.
  At batch 1 that is ~1 FLOP/byte: deeply memory-bound. Floor per step = bytes read / HBM bandwidth.
  Example estimate: an 8B BF16 model is 16 GB of weights, so one decode step is >= 16 / 4800 s ≈ 3.3 ms on H200.
  Batching B live sequences reuses each weight B times, until the KV cache reads dominate.
  Then only fewer bytes help: FP8 KV cache, shorter answers, prefix caching.
- **Prefill** and **log-prob scoring** process many tokens per weight read: compute-bound.
- **Training** forward + backward on packed rows: compute-bound.
- **Small kernels** have a fixed launch cost. If the CPU cannot launch fast enough, the step is overhead-bound,
  not memory- or compute-bound. Fix: fewer, bigger ops (fusion), CUDA graphs, `torch.compile` on fixed shapes.
- Every speed-up does one of two things: more work per byte (batching, tiling) or fewer bytes
  (fusion, FlashAttention, quantization, contiguous memory layout).
- NVML "GPU util %" only says a kernel was running. A starved GPU can show 100%. Use MFU and MBU instead.

Settings (TARGET; see `configs/h200.json`, `configs/b200.json`):

| knob | setting | why |
|---|---|---|
| colocation | trainer and vLLM/SGLang on the same GPU, taking turns (sleep/wake); weights synced in GPU memory after each step | no second machine, no disk round trip |
| async | v1 synchronous (`max_staleness_steps: 0`); overlap CPU work (verifiers, packing) with decode; later allow 1 step stale | one GPU cannot train and decode at the same time |
| model size | <= 4B full fine-tune (default) or 7-8B LoRA | 8B full fine-tune needs ~16 bytes/param ≈ 128 GB (estimate) before the rollout engine |
| batch | sized by tokens per step (H200: ~0.4M = 32 prompts x 8 samples x ~1.5k); micro-batch = one packed row of 16k-32k tokens; grad accumulation for the rest | fewer padded tokens, bigger matmuls |
| packing | on, with `cu_seqlens` | no padding waste, no cross-sample attention |
| attention | FlashAttention-3 varlen on Hopper; the Blackwell FlashAttention build on B200 | tiles stay on chip, the T x T matrix never hits HBM |
| fused kernels | Liger RMSNorm / RoPE / SwiGLU / fused linear + cross-entropy | fewer HBM round trips; no full logits tensor |
| FP8 | H200: FP8 linears for SFT/DPO with BF16 master weights | half the bytes, 2x peak |
| FP4 | B200: rollouts for the token factory and eval only; not GRPO in v1 | low-precision rollouts drift off-policy; GRPO would need an importance correction |
| GRPO precision | log-prob scoring in the same precision as the rollouts | avoids a hidden policy mismatch |
| CUDA graphs | on for decode (engine default), off for training | decode has many tiny kernels; packed training shapes change every step |
| activation checkpointing | off if it fits; on for 8B LoRA at 32k rows; attention-only as a middle option | costs about one extra forward (~+33% compute): hardware FLOPs go up, MFU does not |

MFU / MBU targets per phase (all TARGET, BF16 unless noted):

| phase | bound | target |
|---|---|---|
| rollout decode | memory | MBU 50-70%, MFU 5-20% |
| rollout prefill | compute | MFU 40-60% |
| log-prob scoring | compute | MFU 40-55% |
| train fwd + bwd | compute | MFU 35-50% on H200; FP8 run 25-40% of FP8 peak |
| verifiers | CPU | GPU idle from verifiers < 5% of wall time |
| weight sync + sleep/wake | copy | < 3% of wall time |
| whole GRPO step | decode-heavy | useful MFU (6·N·train tokens / wall / peak) 15-30% |

How to measure:
- MFU = model FLOPs per second / dense peak. Training FLOPs = 6·N·T, prefill and scoring = 2·N·T,
  decode = 2·N per new token (N = matmul weights incl. the LM head). The attention-score term is reported separately.
  Recomputed activations are not counted in MFU.
- MBU = bytes read per second / HBM peak. Decode bytes = weight bytes + KV-cache bytes read per step.
- `workloads.py` returns `flops_per_step` and `bytes_per_step`; `profile.py` divides by the measured step time.

## 5. Profiling

1. **Per-phase numbers.**
   `python -m recipes.coscientist_posttrain.profile --config recipes/coscientist_posttrain/configs/h200.json`
   prints the top ops table, writes `runs/trace_<workload>.json` (open in Perfetto or chrome://tracing)
   and `runs/summary.jsonl` (seconds/step, tokens/s, TFLOPS, MFU, MBU, peak memory). `runs/` is git-ignored.
   The factories also follow the `scripts/profile_gpu.py` contract from PR #79, so once that lands:
   `python scripts/profile_gpu.py --workload recipes.coscientist_posttrain.workloads:decode_step --nvtx`
   adds NVML memory and power samples.
2. **Timeline.** Put NVTX ranges around `data`, `rollout/prefill`, `rollout/decode`, `verify`, `sync`, `wake`,
   `sleep`, `score`, `fwd`, `bwd`, `opt`. Then `nsys profile -t cuda,nvtx,osrt -o runs/grpo_step <command>`.
   Gaps between ranges are idle GPU time.
3. **Kernels.** Take the top 10 kernels from torch.profiler, then run Nsight Compute on them
   (`ncu --set full -k <kernel>`): DRAM throughput, tensor pipe use and occupancy place each kernel on the roofline.
4. **NVML** for memory and power only, not as a health metric.

Bottleneck checklist (symptom -> fix):

| symptom | likely cause | fix |
|---|---|---|
| gap before `fwd` | data loader | pre-tokenize and pre-pack on CPU; pinned memory; prefetch; `num_workers > 0` |
| long tail at the end of `rollout/decode` | a few long answers | length cap; stream finished groups to verifiers |
| gap after decode | CPU verifiers | process pool with timeouts; verify chunks while decode runs |
| `sync` / `wake` > 3% of wall | weight copy | in-GPU copy, no disk; LoRA: sync the adapter only |
| many kernels < 10 µs, CPU timeline not ahead of GPU | launch overhead | fusion (Liger), CUDA graphs, `torch.compile` with bucketed shapes |
| decode MBU high, MFU low | memory-bound (expected) | bigger batch, FP8 KV cache, shorter answers |
| train MFU low, HBM busy | unfused elementwise ops, padding | packing, fused kernels, FlashAttention |
| peak memory near HBM size | activations | selective checkpointing, smaller rows, LoRA |

## 6. How to run

CPU smoke (no GPU, no downloads; also runs in CI):
```
bash recipes/coscientist_posttrain/launch.sh smoke
```
This runs the tests and profiles the four tiny workloads on CPU.

Stage commands (printed, not run):
```
bash recipes/coscientist_posttrain/launch.sh sft  --config recipes/coscientist_posttrain/configs/h200.json
bash recipes/coscientist_posttrain/launch.sh grpo --config recipes/coscientist_posttrain/configs/b200.json
```
`launch.sh` prints the planned full-scale labbench-posttrain / vLLM commands and the smoke command that works today.
The full-scale flags are the planned interface; they do not exist in labbench-posttrain yet (#66).
Nothing is run or faked.

Profile on a GPU (Colab first, then H200/B200):
```
bash recipes/coscientist_posttrain/launch.sh profile --config recipes/coscientist_posttrain/configs/colab.json --i-have-a-gpu
```
The profile model is a GPT-2 built from the config (no download), a proxy for the base model.

## 7. Measured so far

CPU smoke only. No GPU number has been measured.

Run on 2026-11-18, 8 vCPU Linux box shared with other jobs (load average ~16), torch 2.15.0+cpu, e3nn 0.6.0, fp32.
These numbers only show the path runs end to end. They say nothing about GPU speed.

| what | result |
|---|---|
| smoke test (`tests/test_coscientist_recipe.py`) | 8 passed in 17.6 s |
| `sft_step` (2-layer, d=64 GPT-2; 4 x 64 tokens) | 1.44 s/step, 177 tokens/s |
| `prefill_step` (4 x 64 tokens) | 0.35 s/step, 741 tokens/s |
| `decode_step` (batch 8, ctx 128) | 0.13 s/step, 64 tokens/s |
| `geom_sft_step` (6 example records, geometry soft tokens + LM loss) | 3.48 s/step, 204 tokens/s |
| MFU / MBU | null on CPU (no peak table) |

## 8. Next steps on a GPU

1. Colab GPU: `launch.sh profile --config configs/colab.json --i-have-a-gpu`. Check MFU/MBU are not null and the phases
   land on the right side of the roofline. Record GPU, date and git hash.
2. Pick a base model whose license allows training on its outputs; set `base_model` in the config.
3. Build the full-scale CLI in labbench-posttrain (#66) so the planned commands run.
4. H200: one colocated GRPO step with NVTX ranges, `nsys` timeline and `ncu` on the top kernels (#69).
5. Turn on packing + Liger and compare before/after on the same GPU and config (#70).
6. B200: repeat; try FP8 training and FP4 rollouts for the token factory only.
7. Move numbers from TARGET to MEASURED only with a source file in the repo.

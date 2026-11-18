# Lab cost and setup (shared doc)

Shared by YLM, [ouroboros-ocsr](https://github.com/yaniguan/ouroboros-ocsr) and [WignerFlow](https://github.com/yaniguan/WignerFlow).

> [!WARNING]
> All prices are estimates from Nov 2026. GPU prices move daily. Check the source before you spend.

## Prices (Nov 2026)

| item | price | source |
|---|---|---|
| H100 80GB rent | $1.99/hr (RunPod Community PCIe) to $4.39/hr (RunPod Secure SXM); Vast.ai from ~$1.73/hr | runpod.io/pricing, runpeek.app/guides/cloud-gpu-pricing |
| A100 80GB rent | $1.19-1.39/hr (RunPod Community); $2.79/hr (Lambda 8x SXM) | runpod.io/pricing, runailocal.dev/gpu-prices |
| L4 24GB rent | from $0.44/hr (RunPod Community); median ~$0.80/hr | cirrun.com/gpu/l4.html |
| RTX 4090 rent | ~$0.34/hr (RunPod Community / Vast.ai) | runailocal.dev/gpu-prices |
| Colab Pro | $9.99/mo for 100 compute units (~18 A100-40GB hours); Pro+ $49.99/mo for 600 | colab.research.google.com/signup/pricing, fastero.com/blog/google-colab-pro |
| Used RTX 4090 (card only) | ~$2.4k-3.3k (eBay) | gpusold.com, getpcparts.com |
| RTX 5090 (card only) | ~$5.8k-7.4k (MSRP $1,999) | tomshardware.com, videocardprices.com |
| Jetson Orin Nano Super dev kit | $399 | developer.nvidia.com/embedded/faq |
| Jetson AGX Orin 64GB dev kit | $3,499 (was $1,999 before July 2026) | developer.nvidia.com/embedded/faq |

## Buy vs rent (simple math)

| compare | math | result |
|---|---|---|
| Used 4090 vs rented 4090 | ~$3,000 / $0.34/hr | ~8,800 rented hours (about one year at 24/7) |
| RTX 5090 vs rented 4090 | $5,800-7,400 / $0.34/hr | ~17,000-21,800 rented hours |
| RTX 5090 street vs MSRP | $5,800-7,400 / $1,999 | ~2.9-3.7x MSRP |
| Colab Pro per A100 hour | $9.99 / ~18 hr | ~$0.56/hr (A100-40GB) |
| 100 hr of H100 | 100 x $1.99-4.39 | $199-439 |
| 100 hr of A100 80GB | 100 x $1.19-1.39 | $119-139 |

Takeaway: we will not use a card 24/7 for a year soon. Rent in bursts. A card bought now also costs power, a host PC and resale risk (not priced here).

## Lean setup by phase

| phase | goal | where | notes |
|---|---|---|---|
| 0. Prototype | evals, small tests, data checks | Colab Pro; rented L4 or RTX 4090 | CPU for unit tests and stubs |
| 1. Train | teacher answers; student SFT -> DPO -> GRPO | rented A100 / H100 bursts | Use spot/community pods; checkpoint often; resume on preemption |
| 2. Distill | teacher -> ~1-8B student (smaller if accuracy holds) | rented A100 / H100 | Same eval set for teacher and student |
| 3. Edge | 4-bit student next to the robot | Jetson Orin-class box or one 24GB GPU (L4 / 4090) | Buy hardware only here |

Rules:
- Stop pods when idle. Keep data and checkpoints in cheap storage, not on the pod.
- One run = one config + seed + git hash. No unlogged GPU hours.
- Smoke test on CPU or Colab before renting big GPUs.
- Single-GPU saturation: when renting one H200/B200, keep it busy; see [single_gpu_saturation.md](single_gpu_saturation.md).

## Monthly budget (estimate)

Hours are assumptions for planning, not measured usage.

| phase | items (assumed hours) | estimate / month |
|---|---|---|
| 0. Prototype | Colab Pro $9.99 + L4 40 hr x $0.44 + 4090 40 hr x $0.34 | ~$41 |
| 1-2. Train + distill | A100 80GB 150 hr x $1.19-1.39 + H100 40 hr x $1.99-4.39 + Colab Pro | ~$268-394 |
| 3. Edge | Colab Pro + 4090 20 hr x $0.34; Jetson Orin Nano Super $399 one time | ~$17 + $399 once |

Not priced yet (TBD): storage, data egress, teacher API calls if a hosted teacher is used.

## What NOT to buy yet

- No RTX 5090: street price is ~3x MSRP.
- No local 4090 box: ~8,800 rented hours equal one used card.
- No Jetson AGX Orin 64GB ($3,499) until a 4-bit student does not fit the $399 Orin Nano Super.
- No multi-GPU node or long reserved contract until one full SFT -> DPO -> GRPO run works on rented bursts.
- No paid dataset or license until the commercial-safe data plan is clear (BioProBench is CC-BY-NC-4.0).

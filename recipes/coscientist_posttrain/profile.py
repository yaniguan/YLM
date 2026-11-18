"""Profile workloads: timed steps, then a short torch.profiler run with NVTX ranges on CUDA.

python -m recipes.coscientist_posttrain.profile --config recipes/coscientist_posttrain/configs/cpu_smoke.json

Per workload: prints the top ops table, writes <out>/trace_<workload>.json (chrome trace; open
in chrome://tracing or Perfetto) and appends one line to <out>/summary.jsonl with seconds/step,
tokens/s, achieved TFLOPS and MFU/MBU when a peak is known (null on CPU).
"""
import argparse
import contextlib
import json
import os
import re
import time

import torch

from .workloads import FACTORIES

# Dense peak TFLOPS and HBM TB/s (vendor spec sheets; same H200/B200 rows as PR #79).
# Other GPUs or variants (e.g. H200 NVL): pass --peak-tflops / --peak-tbps.
PEAKS = {
    "A100-SXM4-80GB": {"bf16": 312, "tbps": 2.039},
    "A100": {"bf16": 312, "tbps": 1.555},
    "L4": {"bf16": 121, "tbps": 0.3},
    "H200": {"bf16": 989, "fp8": 1979, "tbps": 4.8},
    "B200": {"bf16": 2250, "fp8": 4500, "fp4": 9000, "tbps": 7.7},
}
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
HERE = os.path.dirname(__file__)


def find_peaks(device, precision):
    if device.type != "cuda":
        return {}
    name = torch.cuda.get_device_name(device)
    row = next((v for k, v in PEAKS.items() if re.search(rf"\b{re.escape(k)}\b", name)), {})
    return {"tflops": row.get(precision), "tbps": row.get("tbps")}


def nvtx(on, name):
    return torch.cuda.nvtx.range(name) if on else contextlib.nullcontext()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def ratio(num, seconds, peak, unit):
    return None if num is None or not peak else num / seconds / (peak * unit)


def run_one(name, sizes, args, cfg, device, out):
    prof_cfg = cfg.get("profile", {})
    precision = args.dtype or prof_cfg.get("dtype", "fp32")
    w = FACTORIES[name](torch=torch, device=device, dtype=DTYPES[precision], **sizes)
    cuda = device.type == "cuda"
    for _ in range(prof_cfg.get("warmup", 2)):
        w["step"]()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    steps = args.steps or prof_cfg.get("steps", 10)
    sync(device)
    t0 = time.perf_counter()
    for i in range(steps):
        with nvtx(cuda, f"{name}/step{i}"):
            w["step"]()
    sync(device)
    sec = (time.perf_counter() - t0) / steps

    acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if cuda else [])
    with torch.profiler.profile(activities=acts, record_shapes=True) as prof:
        for i in range(prof_cfg.get("profile_steps", 3)):
            with nvtx(cuda, f"{name}/profiled{i}"):
                w["step"]()
            sync(device)
    sort = "self_cuda_time_total" if cuda else "self_cpu_time_total"
    print(f"\n== {name} ({device.type}, {precision}) top ops ==")
    print(prof.key_averages().table(sort_by=sort, row_limit=args.top))
    trace = os.path.join(out, f"trace_{name}.json")
    prof.export_chrome_trace(trace)

    peaks = find_peaks(device, precision)
    tflops_peak = args.peak_tflops or peaks.get("tflops")
    tbps_peak = args.peak_tbps or peaks.get("tbps")
    flops, nbytes = w.get("flops_per_step"), w.get("bytes_per_step")
    return {
        "workload": name, "sizes": sizes, "device": torch.cuda.get_device_name(device) if cuda else "cpu",
        "dtype": precision, "torch": torch.__version__, "steps": steps, "seconds_per_step": sec,
        "tokens_per_s": w["items_per_step"] / sec, "flops_per_step": flops,
        "attn_flops_per_step": w.get("attn_flops_per_step"),
        "achieved_tflops": None if flops is None else flops / sec / 1e12,
        "achieved_tbps": None if nbytes is None else nbytes / sec / 1e12,
        "peak_tflops": tflops_peak if cuda else None, "peak_tbps": tbps_peak if cuda else None,
        "mfu": ratio(flops, sec, tflops_peak, 1e12) if cuda else None,
        "mbu": ratio(nbytes, sec, tbps_peak, 1e12) if cuda else None,
        "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9 if cuda else None,
        "trace": trace, "tag": "MEASURED", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=os.path.join(HERE, "configs", "cpu_smoke.json"))
    p.add_argument("--workload", choices=sorted(FACTORIES), help="default: all in config profile.workloads")
    p.add_argument("--steps", type=int)
    p.add_argument("--dtype", choices=sorted(DTYPES))
    p.add_argument("--device", help="default: cuda if available, else cpu")
    p.add_argument("--peak-tflops", type=float, help="dense peak for MFU (overrides the table)")
    p.add_argument("--peak-tbps", type=float, help="HBM peak for MBU (overrides the table)")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out", default=os.path.join(HERE, "runs"))
    args = p.parse_args(argv)
    with open(args.config) as f:
        cfg = json.load(f)
    prof_cfg = cfg.get("profile", {})
    model_kw = {k: v for k, v in prof_cfg.get("model", cfg.get("model", {})).items() if k != "tag"}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out, exist_ok=True)
    names = [args.workload] if args.workload else prof_cfg.get("workloads", ["sft_step"])
    results = []
    for name in names:
        sizes = {**model_kw, **prof_cfg.get("sizes", {}).get(name, {})}
        results.append(run_one(name, sizes, args, cfg, device, args.out))
        with open(os.path.join(args.out, "summary.jsonl"), "a") as f:
            f.write(json.dumps(results[-1]) + "\n")
    print("\nworkload        s/step     tokens/s   TFLOPS     MFU     MBU")
    for r in results:
        fmt = lambda v, s: "null" if v is None else s.format(v)  # noqa: E731
        print(f"{r['workload']:<15} {r['seconds_per_step']:.4f}  {r['tokens_per_s']:>9.0f}   "
              f"{fmt(r['achieved_tflops'], '{:.4f}'):<9}  {fmt(r['mfu'], '{:.3f}'):<6}  {fmt(r['mbu'], '{:.3f}')}")
    print(f"summary: {os.path.join(args.out, 'summary.jsonl')}")
    return results


if __name__ == "__main__":
    main()

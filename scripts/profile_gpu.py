#!/usr/bin/env python3
"""Measure how well a workload saturates one GPU. Writes one JSON line per step.

Per step: items/s, achieved TFLOPS, MFU vs a dense peak table, peak memory,
and NVML GPU / memory utilization (if pynvml or nvidia-ml-py is installed).

Workloads:
  --workload toy                 built-in transformer-like MLP step (fwd + bwd + AdamW)
  --workload pkg.mod:make_fn     your factory; see `load_workload` for the contract

Examples:
  python scripts/profile_gpu.py --dry-run
  python scripts/profile_gpu.py --device cpu --steps 3 --tokens 256 --d-model 128
  python scripts/profile_gpu.py --steps 50 --nvtx --out runs/profile_h200.jsonl
  nsys profile -t cuda,nvtx -o runs/grpo_step python scripts/profile_gpu.py --nvtx

On CPU, MFU and NVML fields are null. Outputs are local only; do not commit them.
"""
import argparse
import importlib
import json
import os
import platform
import sys
import time
from contextlib import contextmanager

# Dense (no sparsity) peak TFLOPS. Sources: nvidia.com/en-us/data-center/h200,
# lenovopress.lenovo.com/lp1944 (H200 SXM); NVIDIA Blackwell datasheet,
# lenovopress.lenovo.com/lp2226 (B200 HGX). Other GPUs: pass --peak-tflops.
PEAKS = {
    "H200": {"bf16": 989.0, "fp8": 1979.0, "fp4": None,
             "hbm_gb": 141, "hbm_tb_s": 4.8, "variant": "H200 SXM"},
    "B200": {"bf16": 2250.0, "fp8": 4500.0, "fp4": 9000.0,
             "hbm_gb": 180, "hbm_tb_s": 7.7, "variant": "B200 HGX (SXM6)"},
}
PRECISIONS = ("bf16", "fp8", "fp4")


def match_gpu(device_name):
    """Map a CUDA device name (e.g. 'NVIDIA H200') to a PEAKS key, or None.

    Whole-word match only, so 'GB200' is not 'B200'. 'H200 NVL' has other
    peaks than H200 SXM and returns None (pass --peak-tflops).
    """
    if not device_name:
        return None
    words = device_name.upper().replace("-", " ").split()
    if "NVL" in words:
        return None
    for key in PEAKS:
        if key in words:
            return key
    return None


def peak_tflops(gpu_key, precision="bf16"):
    """Dense peak TFLOPS for a known GPU + precision, else None."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    entry = PEAKS.get(gpu_key) if gpu_key else None
    return entry.get(precision) if entry else None


def achieved_tflops(flops, seconds):
    if flops is None or not seconds or seconds <= 0:
        return None
    return flops / seconds / 1e12


def mfu(flops, seconds, peak):
    """Model FLOPs utilization = achieved FLOP/s / peak dense FLOP/s (0..1)."""
    got = achieved_tflops(flops, seconds)
    if got is None or not peak:
        return None
    return got / peak


def train_flops_dense(n_params, tokens):
    """Rule of thumb for a dense transformer training step: 6 * N * tokens."""
    return 6.0 * n_params * tokens


def hbm_util(bytes_moved, seconds, gpu_key):
    """Achieved bandwidth as a fraction of peak HBM bandwidth, else None."""
    entry = PEAKS.get(gpu_key) if gpu_key else None
    if bytes_moved is None or not entry or not seconds or seconds <= 0:
        return None
    return bytes_moved / seconds / (entry["hbm_tb_s"] * 1e12)


# ---------- NVML ----------

class NVML:
    """Thin NVML reader. `available` is False if pynvml is missing or init fails."""

    def __init__(self, index=0):
        self.available, self.reason, self._h, self._nv = False, None, None, None
        try:
            import pynvml  # provided by the nvidia-ml-py package
            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
            self._nv = pynvml
            self.available = True
        except Exception as e:  # ImportError, NVMLError, ...
            self.reason = f"nvml unavailable ({type(e).__name__})"

    def sample(self):
        if not self.available:
            return {"nvml": self.reason}
        nv = self._nv
        u = nv.nvmlDeviceGetUtilizationRates(self._h)
        m = nv.nvmlDeviceGetMemoryInfo(self._h)
        return {"nvml": "ok", "gpu_util_pct": u.gpu, "mem_util_pct": u.memory,
                "mem_used_gb": m.used / 1e9, "mem_total_gb": m.total / 1e9}

    def close(self):
        if self.available:
            try:
                self._nv.nvmlShutdown()
            except Exception:
                pass


# ---------- workloads ----------

def toy_workload(torch, device, dtype, tokens=4096, d_model=4096, n_layers=4):
    """Transformer-like MLP stack: per layer x @ W1 (d x 4d) -> GELU -> @ W2 (4d x d).

    Returns (step_fn, items_per_step, flops_per_step, bytes_per_step).
    FLOPs use 6 * N * tokens (matmul params only; fwd 2NT + bwd 4NT).
    """
    layers = []
    for _ in range(n_layers):
        layers += [torch.nn.Linear(d_model, 4 * d_model, bias=False),
                   torch.nn.GELU(),
                   torch.nn.Linear(4 * d_model, d_model, bias=False)]
    model = torch.nn.Sequential(*layers).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(tokens, d_model, device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())

    def step():
        opt.zero_grad(set_to_none=True)
        loss = model(x).float().pow(2).mean()
        loss.backward()
        opt.step()

    return step, tokens, train_flops_dense(n_params, tokens), None


def load_workload(spec, torch, device, dtype):
    """Load 'pkg.module:factory'. factory(torch=, device=, dtype=) must return
    a dict {step, items_per_step, flops_per_step?, bytes_per_step?} or a tuple
    (step, items_per_step[, flops_per_step[, bytes_per_step]]). Unknown FLOPs -> None.
    """
    if ":" not in spec:
        raise ValueError("workload must look like 'package.module:function'")
    mod_name, fn_name = spec.split(":", 1)
    fn = getattr(importlib.import_module(mod_name), fn_name)
    out = fn(torch=torch, device=device, dtype=dtype)
    if isinstance(out, dict):
        return (out["step"], out["items_per_step"],
                out.get("flops_per_step"), out.get("bytes_per_step"))
    out = tuple(out) + (None,) * (4 - len(out))
    return out[:4]


# ---------- main loop ----------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--workload", default="toy", help="'toy' or 'pkg.module:factory'")
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
    p.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"],
                   help="tensor dtype (default: bf16 on GPU, fp32 on CPU)")
    p.add_argument("--precision", default="bf16", choices=PRECISIONS,
                   help="which peak column to use for MFU")
    p.add_argument("--gpu", default=None, choices=sorted(PEAKS),
                   help="force the peak-table row (default: from device name)")
    p.add_argument("--peak-tflops", type=float, default=None,
                   help="override peak dense TFLOPS (any GPU)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--tokens", type=int, default=4096, help="toy: tokens per step")
    p.add_argument("--d-model", type=int, default=4096, help="toy: width")
    p.add_argument("--layers", type=int, default=4, help="toy: MLP blocks")
    p.add_argument("--nvtx", action="store_true", help="wrap steps in NVTX ranges")
    p.add_argument("--out", default="-", help="JSONL path, '-' = stdout")
    p.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    return p.parse_args(argv)


def resolve(args, torch):
    """Pick device, dtype, GPU row and peak. No workload is built here."""
    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    is_cuda = dev.startswith("cuda")
    if is_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available; use --device cpu")
    dtype_name = args.dtype or ("bf16" if is_cuda else "fp32")
    name = torch.cuda.get_device_name(torch.device(dev)) if is_cuda else platform.processor() or "cpu"
    gpu_key = args.gpu or (match_gpu(name) if is_cuda else None)
    peak = args.peak_tflops if args.peak_tflops else (peak_tflops(gpu_key, args.precision) if is_cuda else None)
    return {"device": dev, "is_cuda": is_cuda, "dtype": dtype_name, "device_name": name,
            "gpu_key": gpu_key, "precision": args.precision, "peak_tflops": peak}


@contextmanager
def nvtx_range(torch, name, on):
    if on:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if on:
            torch.cuda.nvtx.range_pop()


def _sync(torch, is_cuda):
    if is_cuda:
        torch.cuda.synchronize()


def run(args, out_stream):
    import torch
    cfg = resolve(args, torch)
    meta = {"type": "meta", "workload": args.workload, **cfg,
            "steps": args.steps, "warmup": args.warmup, "torch": torch.__version__,
            "peak_source": ("--peak-tflops" if args.peak_tflops else
                            (PEAKS[cfg["gpu_key"]]["variant"] if cfg["gpu_key"] else None))}
    if cfg["is_cuda"] and cfg["peak_tflops"] is None:
        meta["note"] = "GPU not in peak table; pass --peak-tflops to get MFU"
    if args.dry_run:
        meta["dry_run"] = True
        out_stream.write(json.dumps(meta) + "\n")
        return [meta]

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg["dtype"]]
    device = torch.device(cfg["device"])
    if args.workload == "toy":
        step, items, flops, nbytes = toy_workload(torch, device, dtype, args.tokens, args.d_model, args.layers)
    else:
        step, items, flops, nbytes = load_workload(args.workload, torch, device, dtype)

    nvml = NVML(device.index or 0) if cfg["is_cuda"] else None
    use_nvtx = args.nvtx and cfg["is_cuda"]
    records = [meta]
    out_stream.write(json.dumps(meta) + "\n")
    if cfg["is_cuda"]:
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    try:
        for i in range(args.warmup + args.steps):
            warm = i < args.warmup
            _sync(torch, cfg["is_cuda"])
            t0 = time.perf_counter()
            with nvtx_range(torch, f"{'warmup' if warm else 'step'}_{i}", use_nvtx):
                step()
            _sync(torch, cfg["is_cuda"])
            dt = time.perf_counter() - t0
            rec = {"type": "step", "step": i, "warmup": warm, "seconds": dt,
                   "items_per_s": items / dt if dt > 0 else None,
                   "achieved_tflops": achieved_tflops(flops, dt),
                   "mfu": mfu(flops, dt, cfg["peak_tflops"]) if cfg["is_cuda"] else None,
                   "hbm_bw_util": hbm_util(nbytes, dt, cfg["gpu_key"]) if cfg["is_cuda"] else None,
                   "peak_mem_gb": (torch.cuda.max_memory_allocated(device) / 1e9
                                   if cfg["is_cuda"] else None)}
            rec.update(nvml.sample() if nvml else {"nvml": "nvml unavailable (cpu)"})
            if not warm:
                times.append(dt)
            records.append(rec)
            out_stream.write(json.dumps(rec) + "\n")
    finally:
        if nvml:
            nvml.close()

    mean_dt = sum(times) / len(times) if times else None
    summary = {"type": "summary", "measured_steps": len(times), "mean_seconds": mean_dt,
               "items_per_s": items / mean_dt if mean_dt else None,
               "achieved_tflops": achieved_tflops(flops, mean_dt),
               "mfu": mfu(flops, mean_dt, cfg["peak_tflops"]) if cfg["is_cuda"] and mean_dt else None,
               "peak_mem_gb": records[-1]["peak_mem_gb"] if len(records) > 1 else None}
    if cfg["is_cuda"] and summary["peak_mem_gb"] is not None:
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        summary["peak_mem_pct_of_hbm"] = 100.0 * summary["peak_mem_gb"] / total
    records.append(summary)
    out_stream.write(json.dumps(summary) + "\n")
    return records


def main(argv=None):
    args = parse_args(argv)
    if args.out == "-":
        run(args, sys.stdout)
    else:
        parent = os.path.dirname(args.out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, "w") as f:
            run(args, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())

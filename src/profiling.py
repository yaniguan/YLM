"""Bounded training traces and timing, with optional GPU telemetry."""
from __future__ import annotations
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import torch


class Profiler:
    def __init__(self, directory, device='cpu', trace_steps=20):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.started = time.perf_counter()
        self.phases = {}
        self.tokens = self.samples = self.steps = 0
        self.utilization = []
        self.stop = threading.Event()
        self.trace_steps = trace_steps
        self.trace = None
        self.thread = None
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            self.thread = threading.Thread(target=self._sample_gpu, daemon=True)
            self.thread.start()
        if trace_steps > 0:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.trace = torch.profiler.profile(activities=activities, record_shapes=True,
                                                profile_memory=True)
            self.trace.start()

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextmanager
    def phase(self, name):
        self._sync()
        start = time.perf_counter()
        try:
            with torch.profiler.record_function(name):
                yield
        finally:
            self._sync()
            self.phases[name] = self.phases.get(name, 0.0) + time.perf_counter() - start

    def step_done(self, tokens, samples):
        self.tokens += tokens
        self.samples += samples
        self.steps += 1
        if self.trace is not None:
            self.trace.step()
            if self.steps >= self.trace_steps:
                self._finish_trace()

    def _finish_trace(self):
        if self.trace is None:
            return
        self.trace.stop()
        self.trace.export_chrome_trace(str(self.directory / 'trace.json'))
        (self.directory / 'key_averages.txt').write_text(self.trace.key_averages().table(
            sort_by='self_cpu_time_total', row_limit=50))
        self.trace = None

    def _sample_gpu(self):
        # Index follows CUDA_VISIBLE_DEVICES when it is a physical index or GPU UUID.
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        logical = self.device.index or 0
        physical = visible[logical] if len(visible) > logical and visible[logical] else str(logical)
        while not self.stop.is_set():
            try:
                output = subprocess.check_output(['nvidia-smi', '-i', physical,
                    '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    timeout=2, text=True)
                value = float(output.strip().splitlines()[0])
                self.utilization.append(value)
                with (self.directory / "gpu_utilization.jsonl").open("a") as stream:
                    stream.write(json.dumps({"elapsed_seconds": time.perf_counter() - self.started,
                                             "utilization_percent": value}) + "\n")
            except Exception as exc:
                self.telemetry_error = str(exc)
            self.stop.wait(1.0)

    def close(self):
        self._finish_trace()
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)
        return self.save()

    def save(self):
        elapsed = time.perf_counter() - self.started
        try:
            import psutil
            rss = psutil.Process().memory_info().rss
        except ImportError:
            import resource
            import sys
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        summary = {'wall_seconds': elapsed, 'tokens': self.tokens, 'samples': self.samples,
                   'steps': self.steps, 'tokens_per_second': self.tokens / max(elapsed, 1e-9),
                   'samples_per_second': self.samples / max(sum(self.phases.values()), 1e-9),
                   'throughput_basis': 'synchronized training phases (includes profiling overhead)',
                   'gpu_telemetry_error': getattr(self, 'telemetry_error', None),
                   'phase_seconds': self.phases,
                   'phase_seconds_per_step': {k: v / max(self.steps, 1) for k, v in self.phases.items()},
                   'cpu_rss_bytes': rss,
                   'cuda_peak_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None,
                   'gpu_utilization_mean': sum(self.utilization) / len(self.utilization) if self.utilization else None,
                   'gpu_utilization_samples': self.utilization}
        (self.directory / 'summary.json').write_text(json.dumps(summary, indent=2))
        return summary

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

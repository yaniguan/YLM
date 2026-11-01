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
    def __init__(self, directory, device='cpu', trace_steps=20, sync_steps=200, smi_timeout=10.0,
                 sample_interval=5.0, flush_interval=60.0, max_summary_samples=1000):
        """sync_steps: only the first sync_steps steps are timed with CUDA synchronization per phase (step-time
        breakdown); later steps run unsynchronized so profiling does not throttle the GPU. Throughput is the
        wall-clock rate over all steps (time between consecutive step_done calls within an epoch)."""
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.started = time.perf_counter()
        self.phases = {}
        self.tokens = self.samples = self.steps = 0
        self.sync_steps = int(sync_steps)
        self.smi_timeout = float(smi_timeout)
        self.telemetry_errors = 0
        self.synced_steps = 0
        self.train_wall = 0.0
        self.wall_samples = 0
        self._last_step_t = None
        self.utilization = []
        # Telemetry is buffered and flushed in batches: one append per flush_interval instead of one
        # open/append per sample, which keeps small-write traffic off network filesystems (Drive FUSE).
        self.sample_interval = float(sample_interval)
        self.flush_interval = float(flush_interval)
        self.max_summary_samples = int(max_summary_samples)
        self._pending = []
        self._pending_lock = threading.Lock()
        self._last_flush = time.perf_counter()
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

    def epoch_start(self):
        """Call at the start of each epoch so validation time is not counted as training time."""
        self._last_step_t = time.perf_counter()

    @contextmanager
    def phase(self, name):
        if self.steps >= self.sync_steps and self.trace is None:
            yield  # unsynchronized steady state
            return
        self._sync()
        start = time.perf_counter()
        try:
            with torch.profiler.record_function(name):
                yield
        finally:
            self._sync()
            self.phases[name] = self.phases.get(name, 0.0) + time.perf_counter() - start

    def step_done(self, tokens, samples):
        now = time.perf_counter()
        if self._last_step_t is not None:
            self.train_wall += now - self._last_step_t
            self.wall_samples += samples
        self._last_step_t = now
        if self.steps < self.sync_steps or self.trace is not None:
            self.synced_steps += 1
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
                    timeout=self.smi_timeout, text=True)
                value = float(output.strip().splitlines()[0])
                self.utilization.append(value)
                with self._pending_lock:
                    self._pending.append({"elapsed_seconds": time.perf_counter() - self.started,
                                          "utilization_percent": value})
                if time.perf_counter() - self._last_flush >= self.flush_interval:
                    self.flush_telemetry()
            except Exception as exc:
                self.telemetry_error = str(exc)
                self.telemetry_errors += 1
            self.stop.wait(self.sample_interval)

    def flush_telemetry(self):
        """Append buffered GPU samples in one write."""
        with self._pending_lock:
            rows, self._pending = self._pending, []
        self._last_flush = time.perf_counter()
        if rows:
            with (self.directory / "gpu_utilization.jsonl").open("a") as stream:
                stream.write("".join(json.dumps(r) + "\n" for r in rows))

    def _bounded_samples(self):
        u = self.utilization
        if len(u) <= self.max_summary_samples:
            return list(u)
        stride = -(-len(u) // self.max_summary_samples)
        return u[::stride]

    def close(self):
        self._finish_trace()
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)
        self.flush_telemetry()
        return self.save()

    def save(self):
        self.flush_telemetry()
        elapsed = time.perf_counter() - self.started
        try:
            import psutil
            rss = psutil.Process().memory_info().rss
        except ImportError:
            import resource
            import sys
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        synced = max(self.synced_steps, 1)
        summary = {'wall_seconds': elapsed, 'tokens': self.tokens, 'samples': self.samples,
                   'steps': self.steps, 'tokens_per_second': self.tokens / max(elapsed, 1e-9),
                   'samples_per_second': self.wall_samples / max(self.train_wall, 1e-9),
                   'step_seconds_mean': self.train_wall / max(self.steps - 1, 1),
                   'throughput_basis': 'wall clock between consecutive training steps (validation excluded); '
                                       f'phase breakdown from the first {self.synced_steps} CUDA-synchronized steps',
                   'samples_per_second_synchronized': (self.samples * synced / max(self.steps, 1)) / max(sum(self.phases.values()), 1e-9),
                   'synchronized_steps': self.synced_steps,
                   'gpu_telemetry_error': getattr(self, 'telemetry_error', None),
                   'gpu_telemetry_errors': self.telemetry_errors,
                   'phase_seconds': self.phases,
                   'phase_seconds_per_step': {k: v / synced for k, v in self.phases.items()},
                   'cpu_rss_bytes': rss,
                   'cuda_peak_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None,
                   'gpu_utilization_mean': sum(self.utilization) / len(self.utilization) if self.utilization else None,
                   'gpu_utilization_samples_total': len(self.utilization),
                   'gpu_utilization_samples': self._bounded_samples()}
        (self.directory / 'summary.json').write_text(json.dumps(summary, indent=2))
        return summary

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

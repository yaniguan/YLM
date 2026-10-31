"""Profiler: synchronized breakdown only for the first sync_steps steps; wall-clock throughput over all steps."""
import json
import time

from src.profiling import Profiler


def test_profiler_sync_window_and_wall_throughput(tmp_path):
    prof = Profiler(tmp_path / "profiling", device="cpu", trace_steps=2, sync_steps=3)
    for _epoch in range(2):
        prof.epoch_start()
        for _ in range(4):
            with prof.phase("forward"):
                time.sleep(0.002)
            prof.step_done(tokens=0, samples=8)
    s = prof.close()
    assert s["steps"] == 8 and s["samples"] == 64
    assert s["synchronized_steps"] == 3
    assert s["samples_per_second"] > 0 and s["step_seconds_mean"] > 0
    assert s["phase_seconds_per_step"]["forward"] >= 0.0015
    saved = json.loads((tmp_path / "profiling" / "summary.json").read_text())
    assert saved["throughput_basis"].startswith("wall clock")
    assert (tmp_path / "profiling" / "trace.json").exists()

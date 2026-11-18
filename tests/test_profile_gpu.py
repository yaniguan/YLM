"""Fast, offline, CPU-only tests for scripts/profile_gpu.py."""
import builtins
import importlib.util
import io
import json
import pathlib
import sys

import pytest

_p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "profile_gpu.py"
_spec = importlib.util.spec_from_file_location("profile_gpu", _p)
pg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pg)


# ---------- peak table + math ----------

def test_peak_table_matches_verified_specs():
    assert pg.peak_tflops("H200", "bf16") == 989.0
    assert pg.peak_tflops("H200", "fp8") == 1979.0
    assert pg.peak_tflops("H200", "fp4") is None  # no FP4 tensor cores on Hopper
    assert pg.peak_tflops("B200", "bf16") == 2250.0
    assert pg.peak_tflops("B200", "fp8") == 4500.0
    assert pg.peak_tflops("B200", "fp4") == 9000.0
    assert pg.PEAKS["B200"]["hbm_gb"] == 180 and pg.PEAKS["H200"]["hbm_gb"] == 141


def test_peak_lookup_unknown_and_bad_precision():
    assert pg.peak_tflops(None) is None
    assert pg.peak_tflops("X999") is None
    with pytest.raises(ValueError):
        pg.peak_tflops("H200", "int4")


@pytest.mark.parametrize("name,key", [
    ("NVIDIA H200", "H200"), ("NVIDIA B200", "B200"), ("nvidia h200 nvl", None), ("NVIDIA GB200", None),
    ("NVIDIA A10G", None), ("", None), (None, None)])
def test_match_gpu(name, key):
    assert pg.match_gpu(name) == key


def test_mfu_math():
    # 989e12 FLOPs in 1 s on an H200 BF16 peak -> MFU 1.0
    assert pg.mfu(989e12, 1.0, 989.0) == pytest.approx(1.0)
    # 6 * 1e9 params * 1e5 tokens = 6e14 FLOPs in 2 s = 300 TFLOPS
    f = pg.train_flops_dense(1e9, 1e5)
    assert f == 6e14
    assert pg.achieved_tflops(f, 2.0) == pytest.approx(300.0)
    assert pg.mfu(f, 2.0, 2250.0) == pytest.approx(300.0 / 2250.0)


def test_mfu_none_paths():
    assert pg.mfu(None, 1.0, 989.0) is None
    assert pg.mfu(1e12, 0.0, 989.0) is None
    assert pg.mfu(1e12, 1.0, None) is None


def test_hbm_util():
    assert pg.hbm_util(4.8e12, 1.0, "H200") == pytest.approx(1.0)
    assert pg.hbm_util(7.7e12, 2.0, "B200") == pytest.approx(0.5)
    assert pg.hbm_util(None, 1.0, "H200") is None
    assert pg.hbm_util(1e9, 1.0, None) is None


# ---------- NVML missing ----------

def test_nvml_missing_path(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pynvml":
            raise ImportError("no pynvml")
        return real_import(name, *a, **k)

    monkeypatch.delitem(sys.modules, "pynvml", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    nv = pg.NVML()
    assert nv.available is False
    s = nv.sample()
    assert s["nvml"].startswith("nvml unavailable")
    nv.close()  # must not raise


# ---------- end-to-end on CPU ----------

def _lines(text):
    return [json.loads(l) for l in text.strip().splitlines()]


def test_dry_run_prints_plan_only():
    buf = io.StringIO()
    pg.run(pg.parse_args(["--dry-run", "--device", "cpu"]), buf)
    recs = _lines(buf.getvalue())
    assert len(recs) == 1 and recs[0]["type"] == "meta" and recs[0]["dry_run"] is True
    assert recs[0]["device"] == "cpu" and recs[0]["peak_tflops"] is None


def test_cpu_toy_run_writes_jsonl(tmp_path):
    out = tmp_path / "sub" / "prof.jsonl"
    rc = pg.main(["--device", "cpu", "--steps", "2", "--warmup", "1",
                  "--tokens", "8", "--d-model", "16", "--layers", "1", "--out", str(out)])
    assert rc == 0
    recs = _lines(out.read_text())
    assert [r["type"] for r in recs] == ["meta", "step", "step", "step", "summary"]
    steps = [r for r in recs if r["type"] == "step"]
    assert steps[0]["warmup"] is True and steps[1]["warmup"] is False
    for r in steps:
        assert r["mfu"] is None and r["peak_mem_gb"] is None
        assert r["nvml"].startswith("nvml unavailable")
        assert r["items_per_s"] > 0 and r["achieved_tflops"] > 0
    s = recs[-1]
    assert s["measured_steps"] == 2 and s["mfu"] is None


def test_user_callable_workload(tmp_path, monkeypatch):
    (tmp_path / "my_wl.py").write_text(
        "def make(torch, device, dtype):\n"
        "    x = torch.ones(4, 4, device=device, dtype=dtype)\n"
        "    return {'step': lambda: (x @ x).sum(), 'items_per_step': 4,\n"
        "            'flops_per_step': 2 * 4 * 4 * 4}\n"
        "def make_tuple(torch, device, dtype):\n"
        "    return (lambda: None, 3)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    buf = io.StringIO()
    pg.run(pg.parse_args(["--device", "cpu", "--steps", "1", "--warmup", "0",
                          "--workload", "my_wl:make"]), buf)
    recs = _lines(buf.getvalue())
    assert recs[1]["achieved_tflops"] is not None and recs[-1]["measured_steps"] == 1

    buf = io.StringIO()
    pg.run(pg.parse_args(["--device", "cpu", "--steps", "1", "--warmup", "0",
                          "--workload", "my_wl:make_tuple"]), buf)
    recs = _lines(buf.getvalue())
    assert recs[1]["achieved_tflops"] is None  # unknown FLOPs stay null


def test_bad_workload_spec():
    import torch
    with pytest.raises(ValueError):
        pg.load_workload("no_colon", torch, torch.device("cpu"), torch.float32)

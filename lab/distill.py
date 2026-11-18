"""Distillation plan for the protocol "brain": teacher -> student -> 4-bit.

Recipe: SFT -> DPO -> GRPO.

    python -m lab.distill --config lab/configs/distill_protocol.json --dry-run

`--dry-run` checks the config and prints the plan. Training is not built here
yet: the SFT/DPO/GRPO loops live in github.com/yaniguan/labbench-posttrain
(posttrain/) and YLM's `src/toolkit/posttrain.py` (SFT/DPO for <vec> answers).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ALLOWED_METHODS = ("sft", "dpo", "grpo")   # in this order
ALLOWED_USES = ("eval_only", "train_and_eval")
NON_COMMERCIAL = ("CC-BY-NC",)


class ConfigError(ValueError):
    pass


def load_config(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(cfg: dict) -> list[str]:
    """Raise ConfigError on hard errors. Return a list of warnings."""
    warnings: list[str] = []
    for key in ("name", "teacher", "student", "data", "stages", "quantize"):
        if key not in cfg:
            raise ConfigError(f"missing '{key}'")

    stages = cfg["stages"]
    if not stages:
        raise ConfigError("no stages")
    methods = [str(s.get("method", "")).lower() for s in stages]
    for m in methods:
        if m not in ALLOWED_METHODS:
            raise ConfigError(f"unknown method '{m}'")
    order = [ALLOWED_METHODS.index(m) for m in methods]
    if order != sorted(order):
        raise ConfigError(f"stages must follow sft -> dpo -> grpo, got {methods}")
    if methods[0] != "sft":
        warnings.append("first stage is not SFT; DPO/GRPO usually start from an SFT checkpoint")

    st = cfg["student"]
    lo, hi = st.get("min_params_b"), st.get("max_params_b")
    if lo is not None and hi is not None and lo > hi:
        raise ConfigError("student min_params_b > max_params_b")

    for d in cfg["data"]:
        use = d.get("use")
        if use not in ALLOWED_USES:
            raise ConfigError(f"data '{d.get('name')}': use must be one of {ALLOWED_USES}")
        if d.get("name") == "LAB-Bench" and use != "eval_only":
            raise ConfigError("LAB-Bench is eval only; do not train on it")
        lic = str(d.get("license", ""))
        if any(lic.startswith(p) for p in NON_COMMERCIAL) or d.get("commercial_ok") is False:
            warnings.append(f"data '{d.get('name')}' is {lic or 'non-commercial'}: research use only, not for a product")
        if lic.startswith("TBD"):
            warnings.append(f"data '{d.get('name')}': license not checked yet")

    q = cfg["quantize"]
    if q.get("bits") not in (4, 8, 16):
        raise ConfigError("quantize.bits must be 4, 8 or 16")
    return warnings


def plan_lines(cfg: dict, warnings: Sequence[str]) -> list[str]:
    st = cfg["student"]
    lines = [
        f"plan: {cfg['name']}  ({cfg.get('status', 'plan only')})",
        f"teacher: {cfg['teacher'].get('model', 'TBD')}",
        f"student: {st.get('model', 'TBD')}  size {st.get('min_params_b', '?')}-{st.get('max_params_b', '?')}B",
        "data:",
    ]
    for d in cfg["data"]:
        lines.append(f"  - {d['name']}: {d['use']}  [{d.get('license', 'license TBD')}]")
    lines.append("stages:")
    for i, s in enumerate(cfg["stages"], 1):
        lines.append(f"  {i}. {s['method'].upper():5s} {s.get('name', '')}: {s.get('source', '')}")
    q = cfg["quantize"]
    lines.append(f"quantize: {q['bits']}-bit -> {', '.join(q.get('targets', []))}")
    ev = cfg.get("eval", {})
    if ev:
        lines.append(f"eval: {ev.get('harness', 'TBD')}; compare {', '.join(ev.get('compare', []))}")
    lines.append("metrics: not run yet")
    for w in warnings:
        lines.append(f"WARNING: {w}")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Distillation plan (SFT -> DPO -> GRPO -> quantize)")
    ap.add_argument("--config", default=str(Path(__file__).parent / "configs" / "distill_protocol.json"))
    ap.add_argument("--dry-run", action="store_true", help="check config and print the plan")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    try:
        warnings = validate(cfg)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1
    print("\n".join(plan_lines(cfg, warnings)))
    if not args.dry_run:
        print("training is not implemented here yet; see labbench-posttrain posttrain/", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Distillation plan for the protocol "brain": teacher -> student -> 4-bit.

Recipe: SFT -> DPO -> GRPO.

    python -m lab.distill --config lab/configs/distill_protocol.json --dry-run
    python -m lab.distill --config lab/configs/distill_commercial.json --dry-run --commercial

`--dry-run` checks the config and prints the plan. When `commercial` is true in
the config or `--commercial` is passed, non-commercial datasets (`CC-BY-NC*`,
`commercial_ok: false`, or `TBD` licenses) and unlabeled items in local JSONL
datasets are rejected with `ConfigError`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from lab import protocol_qa as pqa

ALLOWED_METHODS = ("sft", "dpo", "grpo")   # in this order
ALLOWED_USES = ("eval_only", "train_and_eval")
NON_COMMERCIAL = ("CC-BY-NC", "AGPL", "GPL")
COMMERCIAL_SPDX = {
    "MIT",
    "CC0-1.0",
    "APACHE-2.0",
    "BSD-2-CLAUSE",
    "BSD-3-CLAUSE",
    "CC-BY-4.0",
    "UNLICENSE",
    "0BSD",
    "OWN",
}
REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(ValueError):
    pass


def load_config(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_commercial_safe_license(lic: str, *, use: str = "train_and_eval") -> bool:
    norm_lic = str(lic or "").strip().upper()
    if not norm_lic or norm_lic.startswith("TBD"):
        return False
    if any(norm_lic.startswith(p) for p in NON_COMMERCIAL):
        return False
    if use == "eval_only" and norm_lic == "CC-BY-SA-4.0":
        return True
    return norm_lic in COMMERCIAL_SPDX


def validate_dataset_items_commercial(jsonl_path: Path, dataset_name: str) -> int:
    """Verify every item in a local JSONL file has a valid schema and commercial-safe license."""
    if not jsonl_path.exists():
        raise ConfigError(f"data '{dataset_name}': file not found: {jsonl_path}")
    items = pqa.load_jsonl(jsonl_path)
    if not items:
        raise ConfigError(f"data '{dataset_name}': JSONL file is empty: {jsonl_path}")
    for it in items:
        try:
            pqa.validate_item(it)
        except ValueError as exc:
            raise ConfigError(f"data '{dataset_name}': {exc}") from exc
        item_lic = str(it.get("license", "")).strip()
        if not item_lic:
            raise ConfigError(f"data '{dataset_name}': item '{it.get('id')}' missing 'license' field")
        if it.get("commercial_ok") is False or not is_commercial_safe_license(item_lic):
            raise ConfigError(
                f"data '{dataset_name}': item '{it.get('id')}' license '{item_lic}' is not commercial-safe"
            )
    return len(items)


def validate(cfg: dict, *, commercial: bool = False) -> list[str]:
    """Raise ConfigError on hard errors. Return a list of warnings."""
    warnings: list[str] = []
    enforce_commercial = bool(commercial or cfg.get("commercial", False))

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
        name = str(d.get("name", ""))
        use = d.get("use")
        if use not in ALLOWED_USES:
            raise ConfigError(f"data '{name}': use must be one of {ALLOWED_USES}")
        if name == "LAB-Bench" and use != "eval_only":
            raise ConfigError("LAB-Bench is eval only; do not train on it")
        lic = str(d.get("license", ""))
        non_comm = any(lic.upper().startswith(p) for p in NON_COMMERCIAL) or d.get("commercial_ok") is False
        unverified = not lic.strip() or lic.startswith("TBD")

        if enforce_commercial:
            if non_comm or not is_commercial_safe_license(lic, use=use):
                raise ConfigError(
                    f"data '{name}' is {lic or 'non-commercial'}: blocked in commercial product mode"
                )
            if unverified:
                raise ConfigError(f"data '{name}': license must be verified before commercial run")
            if "path" in d:
                raw_p = Path(d["path"])
                resolved = raw_p if raw_p.is_absolute() else (REPO_ROOT / raw_p)
                validate_dataset_items_commercial(resolved, name)
        else:
            if non_comm:
                warnings.append(f"data '{name}' is {lic or 'non-commercial'}: research use only, not for a product")
            if lic.startswith("TBD"):
                warnings.append(f"data '{name}': license not checked yet")

    q = cfg["quantize"]
    if q.get("bits") not in (4, 8, 16):
        raise ConfigError("quantize.bits must be 4, 8 or 16")
    return warnings


def plan_lines(cfg: dict, warnings: Sequence[str], *, commercial: bool = False) -> list[str]:
    st = cfg["student"]
    mode = "commercial-safe" if (commercial or cfg.get("commercial")) else "research"
    lines = [
        f"plan: {cfg['name']}  ({cfg.get('status', 'plan only')}; mode={mode})",
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
    ap.add_argument("--commercial", action="store_true", help="enforce commercial-safe license checks")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.commercial:
        cfg = {**cfg, "commercial": True}
    try:
        warnings = validate(cfg, commercial=args.commercial)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1
    print("\n".join(plan_lines(cfg, warnings, commercial=args.commercial)))
    if not args.dry_run:
        print("training is not implemented here yet; see labbench-posttrain posttrain/", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

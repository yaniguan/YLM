"""Measure padding vs length-bucketing vs FFD sequence-packing token waste on seq_arm prompts."""
from __future__ import annotations

import json
import math
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"


def approx_token_lengths(items: list[dict]) -> dict[str, list[int]]:
    """Compute token lengths under two tokenizations:
    - `mixed_bpe_plus_1nt`: ~4 chars/token on English prose + 1 token per nucleotide on sequences
    - `char_6mer`: ~4 chars/token on prose + ceil(len/6) 6-mer tokens on sequences
    """
    one_nt: list[int] = []
    six_mer: list[int] = []
    for it in items:
        seq_chars = sum(len(s["seq"]) for s in it.get("sequences", []))
        prose_chars = max(0, len(it["question"]) - seq_chars) + sum(len(c) for c in it.get("choices", []))
        prose_toks = max(1, math.ceil(prose_chars / 4.0))
        seq_6 = sum(math.ceil(len(s["seq"]) / 6.0) for s in it.get("sequences", []))
        one_nt.append(prose_toks + seq_chars)
        six_mer.append(prose_toks + seq_6)
    return {"mixed_bpe_plus_1nt": one_nt, "char_6mer": six_mer}


def eval_batching_schemes(lengths: list[int], batch_size: int = 16, pack_cap: int = 1024, seed: int = 0) -> dict:
    real_tokens = sum(lengths)
    max_len = max(lengths)

    # 1. Fixed global max-length padding
    global_pad_total = len(lengths) * max_len
    global_waste = 1.0 - real_tokens / global_pad_total

    # 2. Random micro-batch dynamic padding (batch_size=16)
    rng = random.Random(seed)
    shuf = lengths[:]
    rng.shuffle(shuf)
    dyn_total = sum(
        len(shuf[i : i + batch_size]) * max(shuf[i : i + batch_size])
        for i in range(0, len(shuf), batch_size)
    )
    dyn_waste = 1.0 - real_tokens / dyn_total

    # 3. Length-sorted bucketing (batch_size=16)
    srt = sorted(lengths)
    bucket_total = sum(
        len(srt[i : i + batch_size]) * max(srt[i : i + batch_size])
        for i in range(0, len(srt), batch_size)
    )
    bucket_waste = 1.0 - real_tokens / bucket_total

    # 4. First-fit-decreasing (FFD) sequence packing into fixed `pack_cap` bins
    cap = max(pack_cap, max_len)
    bins: list[int] = []
    for L in sorted(lengths, reverse=True):
        placed = False
        for b_idx, used in enumerate(bins):
            if used + L <= cap:
                bins[b_idx] += L
                placed = True
                break
        if not placed:
            bins.append(L)
    pack_total = len(bins) * cap
    pack_waste = 1.0 - real_tokens / pack_total

    return {
        "n_items": len(lengths),
        "real_tokens": real_tokens,
        "min_len": min(lengths),
        "median_len": sorted(lengths)[len(lengths) // 2],
        "max_len": max_len,
        "global_pad": {"allocated_tokens": global_pad_total, "waste_fraction": round(global_waste, 4)},
        "dynamic_batch_pad_bs16": {"allocated_tokens": dyn_total, "waste_fraction": round(dyn_waste, 4)},
        "length_bucketed_bs16": {"allocated_tokens": bucket_total, "waste_fraction": round(bucket_waste, 4)},
        "ffd_packed": {"bin_cap": cap, "n_bins": len(bins), "allocated_tokens": pack_total, "waste_fraction": round(pack_waste, 4)},
    }


def compute_and_write() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seq_items = [json.loads(line) for line in (DATA_DIR / "items_seq.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    tok_map = approx_token_lengths(seq_items)
    stats = {name: eval_batching_schemes(lens) for name, lens in tok_map.items()}

    (RESULTS_DIR / "padding_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Padding vs Bucketing vs Sequence-Packing Token Waste (`seq_arm`)",
        "",
        f"Measured on `{len(seq_items)}` sequence-dependent items across all 5 protocol families (`tm_anneal`, `restriction`, `pam`, `ext_gc`, `rna_struct`).",
        "",
        "| Tokenization | Real tokens | Min / Med / Max len | Global pad waste | Dynamic batch (bs=16) waste | Length-bucketed (bs=16) waste | FFD packed (`cap=1024`) waste |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in stats.items():
        lines.append(
            f"| `{name}` | {s['real_tokens']:,} | {s['min_len']} / {s['median_len']} / {s['max_len']} | "
            f"{s['global_pad']['waste_fraction'] * 100:.1f}% | "
            f"{s['dynamic_batch_pad_bs16']['waste_fraction'] * 100:.1f}% | "
            f"{s['length_bucketed_bs16']['waste_fraction'] * 100:.1f}% | "
            f"{s['ffd_packed']['waste_fraction'] * 100:.1f}% ({s['ffd_packed']['n_bins']} bins) |"
        )
    lines.append("")
    (RESULTS_DIR / "padding_stats.md").write_text("\n".join(lines), encoding="utf-8")
    return stats


if __name__ == "__main__":
    print(json.dumps(compute_and_write(), indent=2))

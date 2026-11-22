"""Tests for the DNA/RNA sequence ablation arm (`ablation/seq_arm/`, #84)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ablation.seq_arm import packing_stats

ROOT = Path(__file__).resolve().parents[1]
SEQ_DIR = ROOT / "ablation" / "seq_arm"


def test_committed_seq_arm_splits_and_no_scenario_leakage():
    seq_path = SEQ_DIR / "data" / "items_seq.jsonl"
    txt_path = SEQ_DIR / "data" / "items_matched_text.jsonl"
    assert seq_path.exists() and txt_path.exists()

    seq_items = [json.loads(ln) for ln in seq_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    txt_items = [json.loads(ln) for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(seq_items) == len(txt_items) == 180

    fams = {it["family"] for it in seq_items}
    assert fams == {"tm_anneal", "restriction", "pam", "ext_gc", "rna_struct"}

    train_scen = {it["scenario"] for it in seq_items if it["split"] == "train"}
    test_scen = {it["scenario"] for it in seq_items if it["split"] == "test"}
    assert train_scen.isdisjoint(test_scen)

    # Every scenario has twin_valid (PASS), twin_broken (FAIL), and 4-way mcq
    by_scen: dict[str, dict[str, dict]] = {}
    for it in seq_items:
        by_scen.setdefault(it["scenario"], {})[it["kind"]] = it
    for sc, grp in by_scen.items():
        assert set(grp) == {"twin_valid", "twin_broken", "mcq"}, sc
        assert grp["twin_valid"]["answer"] == "PASS"
        assert grp["twin_broken"]["answer"] == "FAIL"
        assert len(grp["mcq"]["choices"]) == 4


def test_padding_bucketing_and_packing_stats_order():
    seq_items = [
        json.loads(ln)
        for ln in (SEQ_DIR / "data" / "items_seq.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    tok_map = packing_stats.approx_token_lengths(seq_items)
    s = packing_stats.eval_batching_schemes(tok_map["mixed_bpe_plus_1nt"])
    # Bucketing must waste no more than global padding, and FFD packing wastes less than global padding
    assert s["length_bucketed_bs16"]["waste_fraction"] <= s["global_pad"]["waste_fraction"]
    assert s["ffd_packed"]["waste_fraction"] < s["global_pad"]["waste_fraction"]

    assert (SEQ_DIR / "results" / "padding_stats.md").exists()
    summary = json.loads((SEQ_DIR / "results" / "summary.json").read_text(encoding="utf-8"))
    assert summary["arms"]["text_plus_seqtools_facts"]["accuracy"] == 1.0
    assert summary["arms"]["matched_text_stated_values"]["accuracy"] == 1.0
    assert summary["arms"]["text_plus_seqtools_facts"]["accuracy"] > summary["arms"]["text_only"]["mean_accuracy"]
    assert summary["arms"]["frozen_dna_encoder_plus_0p5b_lm"]["status"] == "not_run"

    readme = (SEQ_DIR / "README.md").read_text(encoding="utf-8")
    assert "not run" in readme


def test_seqtools_verifiers_when_bio_deps_installed():
    pytest.importorskip("primer3")
    pytest.importorskip("RNA")
    pytest.importorskip("Bio")
    from ablation.seq_arm import seqtools as st

    # Restriction enzyme agreement between Bio.Restriction and string search
    dna = "ACGTGAATTCACGTAAGCTTGGATCC"
    assert st.cut_sites(dna, list(st.ENZYMES)) == st.cut_sites_naive(dna, list(st.ENZYMES))

    # SpCas9 PAM detection on + and - strands
    spacer = "GAGTCCGAGCAGAAGAAGAA"
    assert st.find_pam_spacers("AAAA" + spacer + "TGG" + "TTTT", spacer) is True
    assert st.find_pam_spacers("AAAA" + spacer + "TTC" + "TTTT", spacer) is False
    assert st.find_pam_spacers(st.revcomp("AAAA" + spacer + "AGG" + "TTTT"), spacer) is True

    # ViennaRNA MFE folding on strong GC hairpin vs unstructured A/C sequence
    _, e_hairpin = st.mfe("GGGGCCCCGGGGCCCCUUCGGGGGCCCCGGGGCCCC")
    _, e_open = st.mfe("AACCAACCAAACCAACCAAACCAACC")
    assert st.needs_denature(e_hairpin) is True and st.needs_denature(e_open) is False

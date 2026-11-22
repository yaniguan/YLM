"""Generate sequence-dependent protocol items (with seeded-error twins) and matched text split.

All labels are computed and verified with `primer3`, `Biopython` (`Bio.Restriction`, `MeltingTemp`),
and `ViennaRNA` (`RNA.fold`) via `ablation/seq_arm/seqtools.py`.
"""
from __future__ import annotations

import json
from pathlib import Path
import random

from ablation.seq_arm import seqtools as st

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def _rand_dna(rng: random.Random, length: int, gc_prob: float = 0.5) -> str:
    out = []
    for _ in range(length):
        if rng.random() < gc_prob:
            out.append(rng.choice("GC"))
        else:
            out.append(rng.choice("AT"))
    return "".join(out)


def _scrub_sites(rng: random.Random, seq: str, enzymes: list[str]) -> str:
    """Mutate recognition sites out of `seq` until zero sites remain."""
    s = seq
    for _ in range(80):
        found = st.cut_sites_naive(s, enzymes)
        any_hit = False
        for enz, positions in found.items():
            motif = st.ENZYMES[enz]
            for pos in positions:
                any_hit = True
                mid = pos + len(motif) // 2
                cur = s[mid]
                rep = next(b for b in "ACGT" if b != cur)
                s = s[:mid] + rep + s[mid + 1 :]
                break
            if any_hit:
                break
        if not any_hit:
            break
    assert all(len(v) == 0 for v in st.cut_sites(s, enzymes).values())
    return s


def build_splits(n_per_family: int = 12, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Return `(seq_items, matched_text_items)`.

    Each scenario produces 3 items (2 seeded-error twin yes/no step-check items + 1 4-way MCQ),
    giving `12 * 5 * 3 = 180` sequence-dependent items and `180` matched text items.
    """
    rng = random.Random(seed)
    seq_items: list[dict] = []
    txt_items: list[dict] = []

    splits = ["train"] * 8 + ["test"] * 4  # 8 train / 4 test scenarios per family (zero scenario leakage)

    # 1. Family: tm_anneal (primer Tm & annealing temperature rule: T_anneal = min(Tm_F, Tm_R) - 3 C)
    for k in range(n_per_family):
        sp = splits[k]
        scen_id = f"tm_{k:02d}"
        high_gc = (k % 2 == 0)
        while True:
            pf = _rand_dna(rng, 20, gc_prob=0.68 if high_gc else 0.32)
            pr = _rand_dna(rng, 20, gc_prob=0.65 if high_gc else 0.35)
            tm_f, tm_r = st.tm(pf), st.tm(pr)
            # Ensure primer3 and Biopython NN agree within 2.5 C so label is unambiguous
            if abs(tm_f - st.tm_biopython(pf)) <= 2.5 and abs(tm_r - st.tm_biopython(pr)) <= 2.5:
                break
        target_ta = round(st.anneal_temp(tm_f, tm_r), 1)
        bad_ta = round(target_ta + (9.0 if high_gc else -9.0), 1)
        seqs = [{"name": "primer F", "seq": pf}, {"name": "primer R", "seq": pr}]
        rule_txt = "Rule: set PCR annealing temperature within 1.5 C of min(Tm_F, Tm_R) - 3.0 C."
        stated = f"Stated values: primer F Tm = {tm_f:.1f} C, primer R Tm = {tm_r:.1f} C."

        for variant, ta_val, gold in (("valid", target_ta, "PASS"), ("broken", bad_ta, "FAIL")):
            base_q = (
                f"{rule_txt}\nprimer F: 5'-{pf}-3'\nprimer R: 5'-{pr}-3'\n"
                f"Step sets annealing temperature to {ta_val:.1f} C. Does this step follow the rule? (PASS/FAIL)"
            )
            match_q = (
                f"{rule_txt}\n{stated}\n"
                f"Step sets annealing temperature to {ta_val:.1f} C. Does this step follow the rule? (PASS/FAIL)"
            )
            seq_items.append({
                "id": f"{scen_id}_{variant}", "scenario": scen_id, "split": sp,
                "family": "tm_anneal", "kind": f"twin_{variant}",
                "sequences": seqs, "question": base_q,
                "choices": ["PASS", "FAIL"], "answer": gold,
            })
            txt_items.append({
                "id": f"{scen_id}_{variant}", "scenario": scen_id, "split": sp,
                "family": "tm_anneal", "kind": f"twin_{variant}",
                "sequences": [], "question": match_q,
                "choices": ["PASS", "FAIL"], "answer": gold,
            })

        opts = [f"{target_ta - 8.0:.1f} C", f"{target_ta:.1f} C", f"{target_ta + 6.0:.1f} C", f"{target_ta + 11.0:.1f} C"]
        seq_items.append({
            "id": f"{scen_id}_mcq", "scenario": scen_id, "split": sp, "family": "tm_anneal", "kind": "mcq",
            "sequences": seqs,
            "question": f"{rule_txt}\nprimer F: 5'-{pf}-3'\nprimer R: 5'-{pr}-3'\nWhich annealing temperature follows the rule?",
            "choices": opts, "answer": "B",
        })
        txt_items.append({
            "id": f"{scen_id}_mcq", "scenario": scen_id, "split": sp, "family": "tm_anneal", "kind": "mcq",
            "sequences": [],
            "question": f"{rule_txt}\n{stated}\nWhich annealing temperature follows the rule?",
            "choices": opts, "answer": "B",
        })

    # 2. Family: restriction (choose non-cutting cloning enzyme; avoid internal cut in insert)
    enz_list = ["EcoRI", "BamHI", "HindIII", "XhoI"]
    for k in range(n_per_family):
        sp = splits[k]
        scen_id = f"res_{k:02d}"
        blocked_enz = enz_list[k % len(enz_list)]
        safe_enz = enz_list[(k + 1) % len(enz_list)]
        flank_l = _scrub_sites(rng, _rand_dna(rng, 45), list(st.ENZYMES))
        flank_r = _scrub_sites(rng, _rand_dna(rng, 45), list(st.ENZYMES))
        insert = _scrub_sites(rng, flank_l, [e for e in st.ENZYMES if e != blocked_enz])
        insert = flank_l + st.ENZYMES[blocked_enz] + flank_r
        insert = (
            _scrub_sites(rng, flank_l, list(st.ENZYMES))
            + st.ENZYMES[blocked_enz]
            + _scrub_sites(rng, flank_r, list(st.ENZYMES))
        )
        sites = st.cut_sites(insert, list(st.ENZYMES))
        assert sites == st.cut_sites_naive(insert, list(st.ENZYMES))
        assert len(sites[blocked_enz]) == 1 and len(sites[safe_enz]) == 0
        seqs = [{"name": "insert", "seq": insert}]
        rule_txt = "Rule: cloning enzyme must cut 0 times inside the insert sequence."
        stated = f"Stated values: {blocked_enz} cuts insert 1 time; {safe_enz} cuts insert 0 times."

        for variant, chosen, gold in (("valid", safe_enz, "PASS"), ("broken", blocked_enz, "FAIL")):
            seq_items.append({
                "id": f"{scen_id}_{variant}", "scenario": scen_id, "split": sp,
                "family": "restriction", "kind": f"twin_{variant}", "sequences": seqs,
                "question": f"{rule_txt}\ninsert: 5'-{insert}-3'\nStep digests insert flanks with {chosen}. Valid? (PASS/FAIL)",
                "choices": ["PASS", "FAIL"], "answer": gold,
            })
            txt_items.append({
                "id": f"{scen_id}_{variant}", "scenario": scen_id, "split": sp,
                "family": "restriction", "kind": f"twin_{variant}", "sequences": [],
                "question": f"{rule_txt}\n{stated}\nStep digests insert flanks with {chosen}. Valid? (PASS/FAIL)",
                "choices": ["PASS", "FAIL"], "answer": gold,
            })

        other_cut1 = enz_list[(k + 2) % len(enz_list)]
        other_cut2 = enz_list[(k + 3) % len(enz_list)]
        insert_mcq = (
            _scrub_sites(rng, _rand_dna(rng, 28), list(st.ENZYMES))
            + st.ENZYMES[blocked_enz]
            + _scrub_sites(rng, _rand_dna(rng, 18), list(st.ENZYMES))
            + st.ENZYMES[other_cut1]
            + _scrub_sites(rng, _rand_dna(rng, 18), list(st.ENZYMES))
            + st.ENZYMES[other_cut2]
            + _scrub_sites(rng, _rand_dna(rng, 24), list(st.ENZYMES))
        )
        sites_m = st.cut_sites(insert_mcq, enz_list)
        assert len(sites_m[safe_enz]) == 0 and all(len(sites_m[e]) >= 1 for e in enz_list if e != safe_enz)
        ans_letter = "ABCD"[enz_list.index(safe_enz)]
        seq_items.append({
            "id": f"{scen_id}_mcq", "scenario": scen_id, "split": sp, "family": "restriction", "kind": "mcq",
            "sequences": [{"name": "insert", "seq": insert_mcq}],
            "question": f"{rule_txt}\ninsert: 5'-{insert_mcq}-3'\nWhich enzyme has 0 internal cut sites?",
            "choices": enz_list, "answer": ans_letter,
        })
        txt_items.append({
            "id": f"{scen_id}_mcq", "scenario": scen_id, "split": sp, "family": "restriction", "kind": "mcq",
            "sequences": [],
            "question": (
                f"{rule_txt}\nStated internal cuts: "
                + ", ".join(f"{e}={len(sites_m[e])}" for e in enz_list)
                + ".\nWhich enzyme has 0 internal cut sites?"
            ),
            "choices": enz_list, "answer": ans_letter,
        })

    # 3. Family: pam (SpCas9 20-nt spacer must be immediately followed 3' by NGG on + or - strand)
    for k in range(n_per_family):
        sp = splits[k]
        scen_id = f"pam_{k:02d}"
        good_spacer = _rand_dna(rng, 20)
        bad_spacer = _rand_dna(rng, 20)
        while bad_spacer == good_spacer:
            bad_spacer = _rand_dna(rng, 20)
        target = (
            _rand_dna(rng, 30)
            + good_spacer + "AGG"
            + _rand_dna(rng, 24)
            + bad_spacer + "ATC"
            + _rand_dna(rng, 26)
        )
        assert st.find_pam_spacers(target, good_spacer) is True
        assert st.find_pam_spacers(target, bad_spacer) is False
        seqs = [
            {"name": "target", "seq": target},
            {"name": "spacer A", "seq": good_spacer},
            {"name": "spacer B", "seq": bad_spacer},
        ]
        rule_txt = "Rule: SpCas9 sgRNA spacer must match target immediately 5' of an NGG PAM."
        stated = "Stated values: spacer A is followed by AGG (valid NGG PAM); spacer B is followed by ATC (no PAM)."

        for variant, label_s, gold in (("valid", "spacer A", "PASS"), ("broken", "spacer B", "FAIL")):
            seq_items.append({
                "id": f"{scen_id}_{variant}", "scenario": scen_id, "split": sp,
                "family": "pam", "kind": f"twin_{variant}", "sequences": seqs,
                "question": (
                    f"{rule_txt}\ntarget: 5'-{target}-3'\nspacer A: {good_spacer}\nspacer B: {bad_spacer}\n"
                    f"Step transfects SpCas9 RNP loaded with {label_s}. Valid? (PASS/FAIL)"
                ),
                "choices": ["PASS", "FAIL"], "answer": gold,
            })
            txt_items.append({
                "id": f"{scen_id}_{variant}", "scenario": scen_id, "split": sp,
                "family": "pam", "kind": f"twin_{variant}", "sequences": [],
                "question": f"{rule_txt}\n{stated}\nStep transfects SpCas9 RNP loaded with {label_s}. Valid? (PASS/FAIL)",
                "choices": ["PASS", "FAIL"], "answer": gold,
            })

        seq_items.append({
            "id": f"{scen_id}_mcq", "scenario": scen_id, "split": sp, "family": "pam", "kind": "mcq",
            "sequences": seqs,
            "question": f"{rule_txt}\ntarget: 5'-{target}-3'\nspacer A: {good_spacer}\nspacer B: {bad_spacer}\nWhich guide cleaves the target?",
            "choices": ["spacer A", "spacer B", "neither guide", "both guides"], "answer": "A",
        })
        txt_items.append({
            "id": f"{scen_id}_mcq", "scenario": scen_id, "split": sp, "family": "pam", "kind": "mcq",
            "sequences": [],
            "question": f"{rule_txt}\n{stated}\nWhich guide cleaves the target?",
            "choices": ["spacer A", "spacer B", "neither guide", "both guides"], "answer": "A",
        })

    # 4. Family: ext_gc (amplicon length -> extension time 60 s/kb rounded up to 15 s, and GC > 65% -> 5% DMSO)
    for k in range(n_per_family):
        sp = splits[k]
        scen_id = f"ext_{k:02d}"
        high_gc = (k % 2 == 0)
        amp_len = 320 if (k % 3 == 0) else 620  # 320 bp -> 30 s; 620 bp -> 45 s
        pf = _rand_dna(rng, 20, gc_prob=0.72 if high_gc else 0.45)
        pr_bind = _rand_dna(rng, 20, gc_prob=0.72 if high_gc else 0.45)
        pr = st.revcomp(pr_bind)
        body = _rand_dna(rng, amp_len - 40, gc_prob=0.76 if high_gc else 0.42)
        template = _rand_dna(rng, 25, 0.45) + pf + body + pr_bind + _rand_dna(rng, 25, 0.45)
        amp_s = st.amplicon_seq(template, pf, pr)
        assert amp_s is not None and len(amp_s) == amp_len
        gc_pct = st.gc(amp_s)
        dmso_req = st.needs_dmso(gc_pct)
        ext_s = st.ext_time(len(amp_s))
        seqs = [{"name": "template", "seq": template}, {"name": "primer F", "seq": pf}, {"name": "primer R", "seq": pr}]
        rule_txt = "Rule: extension time = ceil_15s(60 s * bp / 1000); add 5% DMSO iff amplicon GC > 65%."
        stated = f"Stated values: amplicon length = {len(amp_s)} bp, amplicon GC = {gc_pct:.1f}%."
        good_desc = f"{ext_s} s extension, {'with 5% DMSO' if dmso_req else 'no DMSO'}"
        bad_desc = f"{ext_s} s extension, {'no DMSO' if dmso_req else 'with 5% DMSO'}"

        for variant, step_txt, gold in (("valid", good_desc, "PASS"), ("broken", bad_desc, "FAIL")):
            seq_items.append({
                "id": f"{scen_id}_{variant}", "scenario": scen_id, "split": sp,
                "family": "ext_gc", "kind": f"twin_{variant}", "sequences": seqs,
                "question": (
                    f"{rule_txt}\ntemplate: 5'-{template}-3'\nprimer F: {pf}\nprimer R: {pr}\n"
                    f"Step uses {step_txt}. Valid? (PASS/FAIL)"
                ),
                "choices": ["PASS", "FAIL"], "answer": gold,
            })
            txt_items.append({
                "id": f"{scen_id}_{variant}", "scenario": scen_id, "split": sp,
                "family": "ext_gc", "kind": f"twin_{variant}", "sequences": [],
                "question": f"{rule_txt}\n{stated}\nStep uses {step_txt}. Valid? (PASS/FAIL)",
                "choices": ["PASS", "FAIL"], "answer": gold,
            })

        choices = [
            "30 s, no DMSO",
            "30 s, with 5% DMSO",
            "45 s, no DMSO",
            "45 s, with 5% DMSO",
        ]
        gold_str = f"{ext_s} s, {'with 5% DMSO' if dmso_req else 'no DMSO'}"
        ans_l = "ABCD"[choices.index(gold_str)]
        seq_items.append({
            "id": f"{scen_id}_mcq", "scenario": scen_id, "split": sp, "family": "ext_gc", "kind": "mcq",
            "sequences": seqs,
            "question": f"{rule_txt}\ntemplate: 5'-{template}-3'\nprimer F: {pf}\nprimer R: {pr}\nSelect the compliant cycling setup:",
            "choices": choices, "answer": ans_l,
        })
        txt_items.append({
            "id": f"{scen_id}_mcq", "scenario": scen_id, "split": sp, "family": "ext_gc", "kind": "mcq",
            "sequences": [],
            "question": f"{rule_txt}\n{stated}\nSelect the compliant cycling setup:",
            "choices": choices, "answer": ans_l,
        })

    # 5. Family: rna_struct (ViennaRNA MFE <= -15 kcal/mol requires 70 C 5-min pre-denaturation)
    for k in range(n_per_family):
        sp = splits[k]
        scen_id = f"rna_{k:02d}"
        strong_stem = (k % 2 == 0)
        if strong_stem:
            arm = _rand_dna(rng, 16, gc_prob=0.80).replace("T", "U")
            rc_arm = st.revcomp(arm.replace("U", "T")).replace("T", "U")
            rna_seq = "AAU" + arm + "UUCG" + rc_arm + "UAA"
        else:
            rna_seq = "".join(rng.choice("ACU") for _ in range(39))
        dotb, energy = st.mfe(rna_seq)
        denat = st.needs_denature(energy)
        assert denat is strong_stem, f"Unexpected MFE {energy} for strong_stem={strong_stem}"
        seqs = [{"name": "RNA", "seq": rna_seq}]
        rule_txt = "Rule: heat-denature RNA at 70 C for 5 min before RT iff 37 C MFE <= -15.0 kcal/mol."
        stated = f"Stated values: RNA MFE = {energy:.1f} kcal/mol ({dotb})."
        good_step = "70 C 5 min pre-denaturation before RT" if denat else "direct RT setup on ice without 70 C heat step"
        bad_step = "direct RT setup on ice without 70 C heat step" if denat else "70 C 5 min pre-denaturation before RT"

        for variant, st_desc, gold in (("valid", good_step, "PASS"), ("broken", bad_step, "FAIL")):
            seq_items.append({
                "id": f"{scen_id}_{variant}", "scenario": scen_id, "split": sp,
                "family": "rna_struct", "kind": f"twin_{variant}", "sequences": seqs,
                "question": f"{rule_txt}\nRNA: 5'-{rna_seq}-3'\nStep performs {st_desc}. Valid? (PASS/FAIL)",
                "choices": ["PASS", "FAIL"], "answer": gold,
            })
            txt_items.append({
                "id": f"{scen_id}_{variant}", "scenario": scen_id, "split": sp,
                "family": "rna_struct", "kind": f"twin_{variant}", "sequences": [],
                "question": f"{rule_txt}\n{stated}\nStep performs {st_desc}. Valid? (PASS/FAIL)",
                "choices": ["PASS", "FAIL"], "answer": gold,
            })

        choices_r = [
            "Heat-denature at 70 C for 5 min before RT",
            "Proceed directly to RT on ice without heating",
            "Boil in 1 M NaOH for 10 min before RT",
            "Freeze-thaw 5 times in water",
        ]
        seq_items.append({
            "id": f"{scen_id}_mcq", "scenario": scen_id, "split": sp, "family": "rna_struct", "kind": "mcq",
            "sequences": seqs,
            "question": f"{rule_txt}\nRNA: 5'-{rna_seq}-3'\nWhich sample preparation step follows the rule?",
            "choices": choices_r, "answer": "A" if denat else "B",
        })
        txt_items.append({
            "id": f"{scen_id}_mcq", "scenario": scen_id, "split": sp, "family": "rna_struct", "kind": "mcq",
            "sequences": [],
            "question": f"{rule_txt}\n{stated}\nWhich sample preparation step follows the rule?",
            "choices": choices_r, "answer": "A" if denat else "B",
        })

    return seq_items, txt_items


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seq_items, txt_items = build_splits(n_per_family=12, seed=0)
    (DATA_DIR / "items_seq.jsonl").write_text(
        "\n".join(json.dumps(r) for r in seq_items) + "\n", encoding="utf-8"
    )
    (DATA_DIR / "items_matched_text.jsonl").write_text(
        "\n".join(json.dumps(r) for r in txt_items) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(seq_items)} sequence items and {len(txt_items)} matched-text items to {DATA_DIR}")


if __name__ == "__main__":
    main()

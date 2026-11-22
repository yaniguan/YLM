"""Deterministic sequence verifiers: the tool layer and the ground truth for item labels."""
import math

import primer3
import RNA
from Bio import Restriction
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp
from Bio.SeqUtils import gc_fraction

ENZYMES = {
    "EcoRI": "GAATTC", "BamHI": "GGATCC", "HindIII": "AAGCTT", "XhoI": "CTCGAG",
    "NotI": "GCGGCCGC", "XbaI": "TCTAGA", "PstI": "CTGCAG", "SalI": "GTCGAC",
    "KpnI": "GGTACC", "NcoI": "CCATGG",
}

_COMP = str.maketrans("ACGTUN", "TGCAAN")


def revcomp(seq):
    return seq.upper().translate(_COMP)[::-1]


# ---------------------------------------------------------------- Tm and GC
def tm(seq):
    return float(primer3.calc_tm(seq, mv_conc=50, dv_conc=1.5, dntp_conc=0.6, dna_conc=250))


def tm_biopython(seq):
    return float(MeltingTemp.Tm_NN(Seq(seq), Na=50, Mg=1.5, dNTPs=0.6, dnac1=250, dnac2=0))


def gc(seq):
    return 100.0 * gc_fraction(seq)


# ---------------------------------------------------------------- restriction
def cut_sites(seq, enzymes):
    """Recognition-site start positions (0-based) per enzyme, found with Bio.Restriction (linear)."""
    batch = Restriction.RestrictionBatch(list(enzymes))
    found = batch.search(Seq(seq), linear=True)
    out = {}
    for enz, cuts in found.items():
        # Bio.Restriction reports the 1-based position of the first base after the top-strand cut.
        out[str(enz)] = sorted(p - 1 - enz.fst5 for p in cuts)
    return {name: out[name] for name in enzymes}


def cut_sites_naive(seq, enzymes):
    """Same as cut_sites, by plain string search of the site on both strands."""
    seq = seq.upper()
    out = {}
    for name in enzymes:
        hits = set()
        for site in {ENZYMES[name], revcomp(ENZYMES[name])}:
            start = seq.find(site)
            while start != -1:
                hits.add(start)
                start = seq.find(site, start + 1)
        out[name] = sorted(hits)
    return out


# ---------------------------------------------------------------- CRISPR PAM
def spacer_context(target, spacer):
    """All occurrences of spacer on both strands: list of (strand, pos_on_that_strand, next_3_nt)."""
    hits = []
    for strand, s in (("+", target.upper()), ("-", revcomp(target))):
        start = s.find(spacer)
        while start != -1:
            hits.append((strand, start, s[start + len(spacer):start + len(spacer) + 3]))
            start = s.find(spacer, start + 1)
    return hits


def find_pam_spacers(target, spacer):
    """True if spacer occurs on either strand immediately followed (3') by NGG (SpCas9)."""
    return any(len(t) == 3 and t[1:] == "GG" for _, _, t in spacer_context(target, spacer))


# ---------------------------------------------------------------- PCR
def amplicon(template, fwd, rev):
    """Product length if fwd matches the + strand and revcomp(rev) matches downstream, else None."""
    template = template.upper()
    i = template.find(fwd)
    if i == -1:
        return None
    j = template.find(revcomp(rev), i + len(fwd))
    if j == -1:
        return None
    return j + len(rev) - i


def amplicon_seq(template, fwd, rev):
    n = amplicon(template, fwd, rev)
    if n is None:
        return None
    i = template.upper().find(fwd)
    return template.upper()[i:i + n]


# ---------------------------------------------------------------- RNA
def mfe(rna):
    structure, energy = RNA.fold(rna)  # ViennaRNA default parameters, 37 C
    return structure, float(energy)


# ---------------------------------------------------------------- stated rules
def anneal_temp(tm_f, tm_r):
    return min(tm_f, tm_r) - 3.0


def ext_time(length_bp):
    """60 s per kb, minimum 15 s, rounded up to the next 15 s."""
    seconds = 60.0 * length_bp / 1000.0
    return max(15, int(math.ceil(seconds / 15.0 - 1e-9)) * 15)


def needs_dmso(gc_percent):
    return gc_percent > 65.0


def needs_denature(mfe_kcal):
    return mfe_kcal <= -15.0


# ---------------------------------------------------------------- facts block
def facts_block(item):
    """Raw tool outputs for the sequences of an item (never the answer or the rule)."""
    seqs = {s["name"]: s["seq"] for s in item["sequences"]}
    fam = item["family"]
    facts = []
    if fam == "tm_anneal":
        for name in ("primer F", "primer R"):
            facts.append(f"{name} Tm {tm(seqs[name]):.1f} C (primer3), GC {gc(seqs[name]):.1f}%")
    elif fam == "restriction":
        sites = cut_sites(seqs["insert"], list(ENZYMES))
        for enz, pos in sites.items():
            where = (" at " + ",".join(str(p) for p in pos)) if pos else ""
            facts.append(f"{enz} sites in insert: {len(pos)}{where}")
    elif fam == "pam":
        target = seqs["target"]
        spacers = [(n, s) for n, s in seqs.items() if n.startswith("spacer")]
        spacers += [(f"step spacer {s}", s) for s in item.get("verifier", {}).get("step_spacers", [])]
        for name, sp in spacers:
            hits = spacer_context(target, sp)
            if not hits:
                facts.append(f"{name}: not found in target")
            for strand, pos, nxt in hits:
                facts.append(f"{name}: found on {strand} strand at {pos}, followed by {nxt or 'end'}")
    elif fam == "ext_gc":
        amp = amplicon_seq(seqs["template"], seqs["primer F"], seqs["primer R"])
        facts.append(f"amplicon {len(amp)} bp")
        facts.append(f"amplicon GC {gc(amp):.1f}%")
    elif fam == "rna_struct":
        for name, s in seqs.items():
            structure, energy = mfe(s)
            facts.append(f"{name} MFE {energy:.1f} kcal/mol (ViennaRNA, 37 C), dot-bracket {structure}")
    return "Tool facts: " + "; ".join(facts) + "."

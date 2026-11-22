# DNA/RNA Sequence Arm (`ablation/seq_arm/`)

Tests whether explicit DNA/RNA sequence analysis changes accuracy on sequence-dependent wet-lab protocol tasks (`tm_anneal`, `restriction`, `pam`, `ext_gc`, `rna_struct`).

## Dataset splits (`data/`)

- **`data/items_seq.jsonl`** (`180` items = `60` scenarios × `3` items: valid twin, seeded-error broken twin, and 4-way MCQ; `120` train / `60` held-out scenario test). Ground-truth labels are verified with `primer3`, `Biopython` (`Bio.Restriction`, `MeltingTemp`), and `ViennaRNA` (`RNA.fold`).
- **`data/items_matched_text.jsonl`** (`180` items): identical scenarios and answers, with raw nucleotide strings replaced by stated physical values (`Tm`, cut count, PAM status, amplicon length/GC%, or 37 °C MFE).

## Measured CPU results (`results/summary.json`)

Held-out test split (`n = 60`, zero scenario overlap with train; random-guess baseline = `0.417`):

| Arm | Test accuracy | Notes |
|---|---|---|
| `text_only` (sequence masked, 3 seeds) | `0.494 ± 0.008` | Prompt words without sequence tools stay near chance |
| `kmer_4` (raw 4-mer + GC/length probe, 3 seeds) | `0.511 ± 0.016` | Picks up GC/Tm trend, misses exact PAM/cut/amplicon boundaries |
| `shuffled_seq` control (base-shuffled, 3 seeds) | `0.511 ± 0.016` | Destroys motif order (`GAATTC`, `NGG`, hairpins) while keeping GC% |
| `text + seqtools facts` (`seqtools.facts_block`) | `1.000` | Exact `primer3` / `Bio.Restriction` / `ViennaRNA` tool facts resolve all 5 families |
| `matched_text` split (key values stated in words) | `1.000` | Positive control confirming errors come from missing sequence facts |
| `text + frozen DNA encoder -> 0.5B LM` | `not run` | Requires external Hugging Face DNA + Qwen2.5-0.5B weights |

## Padding vs bucketing vs packing waste (`results/padding_stats.md`)

Under `mixed_bpe_plus_1nt` tokenization (`83`–`766` tokens/item):
- Global max-length padding wastes **68.8%** of allocated tokens.
- Dynamic random micro-batch padding (`bs=16`) wastes **67.0%**.
- Length-bucketed batching (`bs=16`) cuts waste to **10.4%**.
- First-fit-decreasing sequence packing (`cap=1024`) cuts waste to **4.6%** (`44` packed bins).

See [`LITERATURE.md`](LITERATURE.md) for audited DNA/RNA foundation model sizes and licenses, and [`results/padding_stats.md`](results/padding_stats.md) for both `1nt` and `6-mer` tables.

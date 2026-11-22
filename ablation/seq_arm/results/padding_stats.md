# Padding vs Bucketing vs Sequence-Packing Token Waste (`seq_arm`)

Measured on `180` sequence-dependent items across all 5 protocol families (`tm_anneal`, `restriction`, `pam`, `ext_gc`, `rna_struct`).

| Tokenization | Real tokens | Min / Med / Max len | Global pad waste | Dynamic batch (bs=16) waste | Length-bucketed (bs=16) waste | FFD packed (`cap=1024`) waste |
|---|---|---|---|---|---|---|
| `mixed_bpe_plus_1nt` | 42,966 | 83 / 133 / 766 | 68.8% | 67.0% | 10.4% | 4.6% (44 bins) |
| `char_6mer` | 14,460 | 51 / 60 / 176 | 54.4% | 51.7% | 5.1% | 5.9% (15 bins) |

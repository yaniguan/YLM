# Literature notes: DNA/RNA models, variable-length batching, sequence + text models

Checked on 2026-11-18. Paper ids were checked against the arXiv API or Crossref. Weights licenses are the Hugging Face license tags; code licenses are the GitHub license fields. Parameter counts come from the Hugging Face safetensors metadata when it is available. Fields marked "not checked" were not verified.

## DNA/RNA foundation models

| Model | Paper | Checkpoint checked | Params | Tokens | Code license | Weights license |
|---|---|---|---|---|---|---|
| Nucleotide Transformer v2 | Nature Methods 2024, doi:10.1038/s41592-024-02523-z | nucleotide-transformer-v2-50m / 500m-multi-species | 55.9M / 498M | 6-mer | CC-BY-NC-SA-4.0 | CC-BY-NC-SA-4.0 |
| Nucleotide Transformer v1 | same | nucleotide-transformer-2.5b-multi-species | 2.5B (from name) | 6-mer | CC-BY-NC-SA-4.0 | CC-BY-NC-SA-4.0 |
| Evo | bioRxiv doi:10.1101/2024.02.27.582234 | evo-1-131k-base | 6.45B | single nucleotide | Apache-2.0 | Apache-2.0 |
| Evo 2 | bioRxiv 2025 (not checked) | evo2_1b_base / 7b / 40b | 1B / 7B / 40B (from names) | single nucleotide | Apache-2.0 | Apache-2.0 |
| DNABERT-2 | arXiv:2306.15006 | DNABERT-2-117M | 117M (counted locally) | BPE | Apache-2.0 | no tag on the model card |
| HyenaDNA | arXiv:2306.15794 | hyenadna-tiny-1k / large-1m | 0.45M / 54.6M | single nucleotide | Apache-2.0 | BSD-3-Clause |
| Caduceus | arXiv:2403.03234 | caduceus-ps seqlen-131k d256 L16 | 7.7M | single nucleotide | Apache-2.0 | Apache-2.0 |
| RNA-FM | arXiv:2204.00300 | rnafm (third-party re-upload) | 99.5M | single nucleotide | MIT | re-upload tagged AGPL-3.0; use the original release |
| RiNALMo | arXiv:2403.00043 | not checked | not checked | single nucleotide | Apache-2.0 | not checked |
| GENA-LM | not checked | gena-lm-bert-base-t2t | 135M | BPE | MIT | no tag on the model card |

None of these models was trained on protocol text.

## Variable-length sequences: padding, bucketing, packing

- Krell et al., "Efficient Sequence Packing without Cross-contamination", arXiv:2107.02027. Packing with per-sequence attention masks and position resets.
- Zhao et al., "Analysing The Impact of Sequence Composition on Language Model Pre-Training", arXiv:2402.13991. Masking across packed documents.
- Ding et al., "Fewer Truncations Improve Language Modeling", arXiv:2404.10830. Best-fit packing.
- Kundu et al., "Enhancing Training Efficiency Using Packing with Flash Attention", arXiv:2407.09105. Packing with position ids and varlen attention.
- Wang et al., "Packing Analysis: Packing Is More Appropriate for Large Models or Datasets in Supervised Fine-tuning", arXiv:2410.08081.
- Llama 3 report, arXiv:2407.21783. Uses a mask that blocks attention across documents.

Our own padding-waste counts for this item set are in `results/padding_stats.md`.

## Sequence + text models

| Model | Paper | How sequence enters the LM | Weights license |
|---|---|---|---|
| ChatNT | Nature Machine Intelligence 2025, doi:10.1038/s42256-025-01047-1 | frozen DNA encoder + projector into a text LM | no tag on the model card; the code repo is CC-BY-NC-SA-4.0 |
| BioReason | arXiv:2505.23579 | DNA model embeddings + projector into a text LM | not checked |
| BioT5 / BioT5+ | arXiv:2310.07276 / arXiv:2402.17810 | raw tokens in an extended vocabulary | MIT (252M base) |
| Galactica | arXiv:2211.09085 | raw characters inside special DNA / protein tokens | CC-BY-NC-4.0 |
| Omni-DNA | arXiv:2502.03499 | DNA model with an extended text/task vocabulary | not checked |
| NatureLM | arXiv:2502.07527 | one vocabulary with prefixes per entity type | not checked |

We found no paper that tests any of these models on protocol QA, step error detection, or step ordering.

## Protocol benchmarks

| Benchmark | Paper | Sequence-dependent part | License |
|---|---|---|---|
| LAB-Bench | arXiv:2407.10362 | SeqQA (600 public items), CloningScenarios (33 public items); ProtocolQA (108) does not need sequences | CC-BY-SA-4.0 (eval only, has a canary) |
| BioProBench | arXiv:2505.07889 | mostly not sequence-dependent | CC-BY-NC-4.0 |
| BioPlanner | arXiv:2310.10632 | not sequence-dependent | not checked |

Tool use on sequence questions: Aviary (arXiv:2412.21154) and CRISPR-GPT (arXiv:2404.18021) use sequence tools inside agents.

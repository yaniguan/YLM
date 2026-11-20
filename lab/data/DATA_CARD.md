# Data card: Commercial-Safe Protocol Dataset (`commercial_protocols_v1`)

## Purpose

`BioProBench` is licensed under `CC-BY-NC-4.0` (non-commercial research only) and `LAB-Bench` (`CC-BY-SA-4.0`) is evaluation-only with canary strings. Product and commercial student distillation runs (`lab/configs/distill_commercial.json`) need training and validation items whose licenses explicitly allow commercial use.

## Audited protocol data sources

| Source | License | Commercial use | Role in YLM `lab/` | Notes |
|---|---|---|---|---|
| `YLM commercial_protocols_v1` (`lab/data/commercial_protocols_v1.jsonl`) | `MIT` | Yes | `train_and_eval` | Hand-written wet-lab protocol QA, calculation, step-debug, and step-order items in `lab/protocol_qa.py` format |
| `Lab Token Factory` (`scripts/41_lab_token_factory.py`) | `MIT` | Yes | `train_and_eval` | Synthetic LabLean-checked protocols and dual-DB SQL records |
| Wet Lab Protocols corpus (Kulkarni et al., ACL WNUT 2018) | `MIT` | Yes | `train_and_eval` | Annotated wet-lab procedures; check upstream repository license file on import |
| USPTO chemical reactions (Lowe, Figshare 2017) | `CC0-1.0` | Yes | `train_and_eval` | Public-domain patent reaction procedures |
| `CALiSol-23` electrolyte formulations | `CC-BY-4.0` | Yes (with attribution) | `train_and_eval` | Used in `ablation/protocol_factors/` |
| `protocols.io` open-access CC-BY subset (e.g. Addgene / iGEM) | `CC-BY-4.0` | Yes (with attribution) | `train_and_eval` | Verify per-protocol license field (`CC-BY-4.0` vs `CC-BY-NC`) before ingest |
| `LAB-Bench` (`futurehouse/lab-bench`) | `CC-BY-SA-4.0` | Share-alike; eval-only | `eval_only` | Contains canary string; never train on it |
| `BioProBench` (`BioProBench/BioProBench`) | `CC-BY-NC-4.0` | **No** | Research only | Blocked automatically when `commercial: true` in `lab/distill.py` |

## Schema (`lab/data/commercial_protocols_v1.jsonl`)

Each line is one JSON object compatible with `lab/protocol_qa.py`, plus provenance and license fields enforced by `lab/distill.py`:

- `id`: unique item identifier (`comm-mcq-*`, `comm-exact-*`, `comm-order-*`)
- `task`: `"mcq"` | `"exact"` | `"order"`
- `domain`: `"pcr"` | `"dilution"` | `"centrifuge"` | `"reagents"` | `"electrolyte"` | `"cloning"`
- `question`: plain-English protocol question, step-debug prompt, or unordered step list
- `choices`: list of options (`mcq` only)
- `answer`: gold letter (`mcq`), exact value (`exact`), or ordered step IDs (`order`)
- `molecules`: optional list of SMILES strings for `lab/molecule_context.py`
- `license`: SPDX license identifier (`"MIT"`)
- `commercial_ok`: `true`
- `origin`: `"ylm_hand_authored"`

## Checks

Run offline validation and dry-run checks:

```bash
python -m lab.distill --config lab/configs/distill_commercial.json --dry-run --commercial
python -m pytest -q tests/test_lab_scaffold.py
```

# lab/ — lab-scale scaffold

Toy code for the plan in [docs/lab_scale_plan.md](../docs/lab_scale_plan.md). Stdlib only. No model is trained here and no metric is reported.

| file | what it does |
|---|---|
| `protocol_qa.py` | Score JSONL predictions (MCQ, exact match, step order). Build prompts with molecule + WignerFlow `[physics]` lines. |
| `molecule_context.py` | Read ouroboros-ocsr context JSONL (or raw `predictions.csv`) into a molecule block for the prompt. |
| `distill.py` | Check a distillation config and print the plan (`--dry-run`). SFT -> DPO -> GRPO only. |
| `configs/distill_protocol.json` | Research plan config. Models are TBD. |
| `configs/distill_commercial.json` | Commercial-clean distillation config using SPDX-allowlisted data. |
| `data/commercial_protocols_v1.jsonl` | Clean-room `CC0-1.0` protocol QA items checked against LabLean rules. |
| `fixtures/` | Hand-written toy items. Not benchmark data. |

```bash
python -m lab.protocol_qa --items lab/fixtures/protocol_qa_tiny.jsonl --print-prompts
python -m lab.distill --dry-run
python -m pytest -q tests/test_lab_scaffold.py
```

Real benchmark loading and RL environments: [labbench-posttrain](https://github.com/yaniguan/labbench-posttrain).

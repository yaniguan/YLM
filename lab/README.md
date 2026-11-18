# lab/ — lab-scale scaffold

Toy code for the plan in [docs/lab_scale_plan.md](../docs/lab_scale_plan.md). Stdlib only. No model is trained here and no metric is reported.

| file | what it does |
|---|---|
| `protocol_qa.py` | Score JSONL predictions (MCQ, exact match, step order). Build prompts with molecule + WignerFlow `[physics]` lines. |
| `molecule_context.py` | Read ouroboros-ocsr context JSONL (or raw `predictions.csv`) into a molecule block for the prompt. |
| `distill.py` | Check a distillation config and print the plan (`--dry-run`). SFT -> DPO -> GRPO only. |
| `configs/distill_protocol.json` | Plan config. Models are TBD. |
| `fixtures/` | Hand-written toy items. Not benchmark data. |

```bash
python -m lab.protocol_qa --items lab/fixtures/protocol_qa_tiny.jsonl --print-prompts
python -m lab.distill --dry-run
python -m pytest -q tests/test_lab_scaffold.py
```

Real benchmark loading and RL environments: [labbench-posttrain](https://github.com/yaniguan/labbench-posttrain).

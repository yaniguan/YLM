"""Lab-scale scaffold for YLM (the "brain" of a lab agent).

Small, standard-library-only pieces:
- `protocol_qa`: JSONL protocol-QA eval stub (MCQ, exact match, step order).
- `molecule_context`: turn OCSR output rows into a molecule context block.
- `distill`: distillation plan (SFT -> DPO -> GRPO -> quantize) with `--dry-run`.

See docs/lab_scale_plan.md. Nothing here trains a model or reports a result.
"""

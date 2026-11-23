# Design Notes & Research Archive

Early design memos, literature checks, and GPU runbooks moved out of the repository root to keep the top-level layout focused on the core code and benchmarks:

- [`00_glossary.md`](00_glossary.md) — Plain-English glossary of spherical harmonics, irreps, and O(3) parity.
- [`01_related_work.md`](01_related_work.md) — Literature survey (GeoMFormer, EquiLLM, NequIP, MACE, Villar et al.).
- [`02_novelty_check.md`](02_novelty_check.md) — Comparison against prior 3D molecular LLM adapters.
- [`03_technical_design.md`](03_technical_design.md) — Mathematical specification of `IrrepsToText` Options A, B, and C.
- [`04_data_and_benchmark.md`](04_data_and_benchmark.md) — `TensorQA-v0` generation and 32-rotation evaluation protocol.
- [`05_risks.md`](05_risks.md) — Failure modes and symmetry-leak checks.
- [`06_plan_12_weeks.md`](06_plan_12_weeks.md) — Original 12-week milestone plan.
- [`HIGHLIGHTS.md`](HIGHLIGHTS.md) — Summary of early benchmark findings.
- [`progress.md`](progress.md) — Chronological engineering log.
- [`waiting4gpu.md`](waiting4gpu.md) — Multi-GPU training commands for queued runs (`Issues #56, #58, #64-#70`).

# Lab token factory: keep only what the checkers confirm

The lab agent proposes tasks, writes protocols or SQL, and writes its own reasoning. A record is kept only if every claim in it passes strict checkers. Kept records are tagged "verified" and can feed post-training (the recipe in PR #86). Everything else is tagged "rejected" with the first checker that failed.

## Loop

1. **Propose.** The agent picks a goal (fill one plate well with 1-4 reagents) or a question about a toy lab database.
2. **Act and reason.** It writes a protocol, then a step-by-step trace: each step's outcome, the volumes it changed, and a final verdict. For SQL it writes the query and the answer.
3. **Check.** Checkers run in order and stop at the first failure.
4. **Keep or reject.** Every record gets provenance tags.
5. **Learn and grow.** The agent remembers which of its claims were refuted. After a rule has caught it 3 times, it stops ignoring that rule. A task level (number of reagents) opens only after the level below reaches 70% verified yield in a round. This is the trusted regime growing.

## Checkers

| Stage | Domain | Check |
|---|---|---|
| V0_format | protocol | ASCII, one line |
| V1_lablean | protocol | LabLean (Lean 4, PR #92) runs the protocol; the agent's verdict must equal Lean's (pass, or the same failing step and error) |
| V2_mirror_agrees | protocol | a Python copy of LabLean (`src/lab_world.py`) must give exactly Lean's result |
| V3_volumes | protocol | every volume the agent claims, step by step, must equal the replay |
| S1-S4 | SQL | the 4-layer gate from lab-verify: parse, read-only, schema, execution match with gold. Execution must also match on a second database with other rows |
| S5_answer | SQL | the stated answer equals the executed result |
| dedup | both | an exact copy of a kept record is rejected |

Lean is called in batches: the protocols are written to one `.lean` file that imports LabLean and calls LabLean's own `step` function, and the file is run with `lake env lean`. About 150 protocols take 8 s.

## Provenance tags (every record)

`id`, `text_sha256`, `source` (agent), `generator`, `factory` version, `domain`, `seed`, `round`, `level`, `checkers` (LabLean git commit and Lean toolchain, Python copy version, SQL gate version), `verdicts` (each checker's pass/fail), `trust` (verified or rejected), `first_fail`, and `data` (synthetic).

## Measured run (CPU only)

`python scripts/41_lab_token_factory.py --lablean_dir lean --seeds 0 1 2` (6 rounds, 200 protocols and 50 SQL tasks per round, 6 min).

| | Protocol | SQL |
|---|---|---|
| records | 3,594 | 900 |
| kept (verified) | 3,187 (88.7%) | 211 (23.4%) |
| checked by Lean | 3,594 | - |
| Python copy disagrees with Lean | 0 | - |

Rejections by first failing checker:

- Protocol: 289 at V3_volumes (wrong claimed volumes) and 118 at V1_lablean (wrong verdict).
- SQL: 360 dedup (there are only 16 question templates), 127 execution, 71 read-only, 54 wrong stated answer, 43 parse and 34 schema.

The trusted regime grows:

- Round-0 protocol yield across the 3 seeds was 153, 109 and 176 out of 200, a mean of 73.0%. By round 5 it was 189, 177 and 189 out of 200, a mean of 92.5%.
- The agent started with 3, 4 and 2 blind spots. It had none left by round 3 for every seed.
- Level 4 opened in round 2 for seeds 0 and 2, and in round 3 for seed 1.
- After round 3, the remaining rejections come from arithmetic slips, which V3 catches.

SQL gate:

- The second database matters. Over 6,400 wrong candidate queries (200 seeds), 259 (4.0%) pass the gate with one database. Only 31 (0.5%) pass with two.
- In the run above, 0 of the 211 kept SQL records were wrong queries.

## Limits

- The agent is a scripted program with mistakes injected on purpose, not a language model. The numbers describe the checkers and the loop, not a model.
- All data is synthetic. LabLean's rules are a small model of a bench (see PR #92 for what it does not cover). A protocol that passes LabLean can still be a bad protocol.
- "Learning" here is a memory of refuted rules, not weight updates.
- Execution match on two databases is still not a proof that a query is right.
- The Lean checks need PR #92's `lean/` built (`lake build`). Without it the Python copy is used alone, and the records say `V1_lablean: skipped`.

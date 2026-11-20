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
| V1_lablean | protocol | LabLean (Lean 4, PR #92) runs the protocol; the agent's verdict must equal Lean's (pass, or the same failing step and error). Without Lean this check is `skipped` and the same comparison against the Python copy is recorded as `V1b_mirror` |
| V2_mirror_agrees | protocol | a Python copy of LabLean (`src/lab_world.py`) must give exactly Lean's result |
| V3_volumes | protocol | every volume the agent claims, step by step, must equal the replay |
| S1-S4 | SQL | the 4-layer gate from lab-verify: parse, read-only, schema, execution match with gold. Execution must also match on a second database with other rows |
| S5_answer | SQL | the stated answer equals the executed result |
| dedup | both | an exact copy of a record already kept (by any seed in the run) is rejected |

Lean is called in batches: the protocols are written to one `.lean` file that imports LabLean and calls LabLean's own `step` function, and the file is run with `lake env lean`. About 150 protocols take 8 s.

## Provenance tags (every record)

`id`, `text_sha256`, `source` (agent), `generator`, `factory` version, `domain`, `seed`, `round`, `level`, `checkers` (LabLean git commit and Lean toolchain, Python copy version, SQL gate version), `verdicts` (each checker's pass/fail), `trust` (verified or rejected), `first_fail`, and `data` (synthetic).

## Measured run (CPU only)

The Lean checks use LabLean from PR #92 at commit `7c17554` (toolchain `leanprover/lean4:v4.36.0`). PR #92 must be merged first; until then `lean/` is not in this branch, the Lean test is skipped, and the Lean numbers below can only be reproduced with a separate checkout of that commit built with `lake build`. There is no CI job with Lean. PR #92 has since moved to `a30571a`, which changes the model (new error cases, rotor balance by mass, a tip that tracks each liquid, amounts times 10,000, and a checked start state). The Lean file that `src/lab_world.py` writes does not compile against `a30571a`, so these numbers are for `7c17554` only. Porting `src/lab_world.py` and its Lean bridge to `a30571a` and re-running is a follow-up.

`python scripts/41_lab_token_factory.py --lablean_dir <LabLean 7c17554>/lean --seeds 0 1 2` (6 rounds, 200 protocols and 50 SQL tasks per round, 106 s on CPU). Dedup is across all seeds. The output is committed as `results/lab_token_factory/stats.json`.

| | Protocol | SQL |
|---|---|---|
| records | 3,594 | 900 |
| kept (verified) | 3,187 (88.7%) | 165 (18.3%) |
| checked by Lean | 3,594 | - |
| Python copy disagrees with Lean | 0 | - |

Rejections by first failing checker:

- Protocol: 289 at V3_volumes (wrong claimed volumes) and 118 at V1_lablean (wrong verdict). Without Lean the same check is recorded as `V1b_mirror`, and `stats.json` writes `mirror_disagrees_with_lean: null`.
- SQL: 406 dedup (there are only 16 question templates, so many questions repeat across rounds and seeds), 127 execution, 71 read-only, 54 wrong stated answer, 43 parse and 34 schema.

An earlier run deduplicated only within each seed and kept 211 SQL records, including copies kept by more than one seed. That run is superseded.

The trusted regime grows:

- Round-0 protocol yield across the 3 seeds was 153, 109 and 176 out of 200, a mean of 73.0%. By round 5 it was 189, 177 and 189 out of 200, a mean of 92.5%. Rounds with levels 1-3 open make 198 protocols, not 200 (66 per level).
- The agent started with 3, 4 and 2 blind spots. It had none left by round 3 for every seed.
- Level 4 opened at the end of round 2 for seeds 0 and 2, and at the end of round 3 for seed 1. So level-4 tasks first appear in round 3 (seeds 0 and 2) and in round 4 (seed 1).
- After round 3, the remaining rejections come from arithmetic slips, which V3 catches.

SQL gate:

- The second database matters. Over 6,400 wrong candidate queries (200 seeds), 259 (4.0%) pass the gate with one database. Only 31 (0.5%) pass with two. The script computes this and writes it to `stats.json` as `sql_lucky_pass` (`--sql_lucky_seeds 200`, the default).
- In the Lean run, 0 of the 165 kept SQL records were wrong queries.

## Limits

- The agent is a scripted program with mistakes injected on purpose, not a language model. The numbers describe the checkers and the loop, not a model.
- All data is synthetic. LabLean's rules are a small model of a bench (see PR #92 for what it does not cover). A protocol that passes LabLean can still be a bad protocol.
- "Learning" here is a memory of refuted rules, not weight updates.
- Execution match on two databases is still not a proof that a query is right.
- The Lean checks need PR #92's `lean/` built (`lake build`). Without it the Python copy is used alone, and the records say `V1_lablean: skipped`.
- Python and Lean agreement is only tested where a built LabLean is present. In this branch and in CI, the Lean test is skipped.

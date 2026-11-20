# Token-quality test: web text vs agent tokens vs verified agent tokens

Question: given the same number of training tokens, do checked agent tokens teach a model more than web text?

## Setup (CPU only)

**Training arms.** Each arm gets exactly D characters, with D = 0.25M or 1M. Tokens are characters.

- **web:** real protocol text written by people, from the Wet Lab Protocols corpus (622 protocols, MIT license).
  The WLP-Dataset protocols come from protocols.io (mostly CC BY 4.0); no corpus text is stored in this repo, only SHA-256 hashes.
- **agent:** every record the lab token factory's agent wrote, unchecked (#97). In the 4,500-record pool, 1,948 records (43.3%) hold a wrong claim: 1,679 (37.3%) a wrong verdict and 269 (6.0%) wrong volumes. In the agent@1M sample the share is 42.7% (845 + 138 of 2,303 records: 36.7% + 6.0%). Counts are in [data_card.json](../results/token_quality_v2/data_card.json) (`agent_pool` and `agent@1000000`, `rejected_by_reason`).
- **verified:** only records that passed LabLean, the Python copy of LabLean, and the volume replay.
- **oracle (positive control):** text in the eval's own question-and-answer format, on other protocols.
- **untrained:** the same model with no training.

The training texts are unchanged from the first run: the SHA-256 of all seven texts in [results/token_quality_v2/data_card.json](../results/token_quality_v2/data_card.json) equal those in [results/token_quality/data_card.json](../results/token_quality/data_card.json). All entries of that card except the two `web@` entries were regenerated with `python scripts/42_token_quality.py --lablean_dir lean --data_card_only --arms agent verified oracle` (no training, Lean not installed, so checks used the Python copy of LabLean, `checked_by`) and came out identical. The web corpus was not available for that check; the `web@` entries are identical to those in the first run's committed card.

**Model.** A tiny GPT with 4 layers, d=128, 4 heads, context 512 and 0.87M parameters. It trains from scratch for 400 steps at batch 8, so it sees 1.64M characters.

## Revised scoring (declared before re-scoring, 2026-11-18)

This section was written before any model was scored on eval v2. Nothing below may be changed after seeing eval v2 results.

### Why the first eval cannot be used

The first eval (v1) failed the positive control for two design reasons, not because of model scale:

1. **The answer prior leaks through the distractors.** Correct answers follow the real error distribution, but distractors were drawn uniformly. A question-free scorer that picks the option seen most often as an answer in the oracle's training documents scores **0.417** on eval v1 (verdict 0.380, volume 0.598; `eval_v1.prior_only_acc` in [data_card.json](../results/token_quality_v2/data_card.json)). That is above the oracle's raw 0.373, so the oracle's raw accuracy is explained by the answer prior, not by reading the question.
2. **The padding dots are scored.** Every option was padded with dots to the longest option's length, and the dots were scored. On eval v1, `pass` gets 22.1 dots on average (17 to 25) and `fail at step k: rule` options 1.7. No training text has dots after an answer. In the committed per-item results, all 20 trained runs get 0 of 82 `pass` items right under raw scoring; the untrained model gets 82 of 82 because it always picks `pass`. Under calibrated scoring, accuracy on the `pass` items swings between seeds of the same arm (agent@250k: 0.01, 1.00, 0.29), which explains much of the seed-to-seed spread.
3. **So calibration removed the prior the oracle had learned.** The calibrated score subtracts log p(option | "A: "), and "A: " is exactly the oracle's answer cue. What remains favours rare options, which are mostly distractors; on items whose answer is a failure, the oracle's calibrated accuracy is 0.22 and 0.28 against raw 0.45 and 0.48. "A: " never appears in the web or agent text, so calibration also treats the arms differently.

Also, 60 of the 482 v1 items were longer than the context and lost the start of the question (`eval_v1.truncated_items`).

The per-item numbers in points 2 and 3 come from joining `results/token_quality/runs.jsonl` with the v1 items rebuilt by `TQ.build_eval_v1` (Python copy of LabLean).

### Eval v2 (`build_eval` in [token_quality.py](../src/token_quality.py))

- **Exchangeable options.** Held-out protocols are grouped in blocks of 4 with 4 different true outcomes: verdict blocks share the number of steps; volume blocks are passing protocols of the same level with 4 different final volumes. The 4 items of a block share one option list in one order, and each option is the true answer of exactly one of them. Any scorer that ignores the question therefore scores exactly 25%. The prior-only scorer above scores 0.25 on eval v2 (`eval.prior_only_acc`).
- **No truncation.** Protocols whose question plus the longest possible option and a newline do not fit in 513 characters are rejected (342 rejected while building, `eval.construction`), and scoring raises an error instead of truncating.
- **No padding**, and no option is a prefix of another option of the same item (asserted).
- Size: 440 items on 400 held-out protocols (400 verdict, 40 volume). `items_sha256` in the data card fixes the items.

### Primary score

**Unpadded likelihood:** pick argmax over options of Σ log p(option | question). No padding, no content-free prompt, no terminator. This is the quantity training optimizes.

### Positive-control rule (gate)

The gate passes only if both hold, on the primary score with the oracle at D = 1M (3 seeds) and the untrained model (3 seeds):

1. the oracle's mean accuracy is higher than the untrained model's, and
2. the paired two-level bootstrap (seeds and items) 95% CI of (oracle − oracle with questions shuffled within level) excludes 0, lying above 0. Shuffled questions come from another item of the same kind and level and from a different block.

If the gate fails, nothing is concluded about the arms, and the arms are not re-run on the same seeds with a changed rule.

### Secondary scores and baselines (always reported, never used for the conclusion)

- old padded raw: Σ log p(option padded with dots | question)
- old padded calibrated: padded raw − Σ log p(option padded with dots | "A: ")
- per-character: Σ log p(option | question) / number of characters
- newline-terminated: Σ log p(option + newline | question)
- prior-only: the question-free majority answer from the oracle's training documents
- the eval v1 calibrated and raw scores on the v1 items, to compare item for item with `results/token_quality/runs.jsonl` (`--compare_v1`); training is seeded and single-threaded, so identical models give identical vectors

Every run saves the per-option log-probs for all of these, so a later scoring change needs no retraining.

### Procedure

1. Gate first: `python scripts/42_token_quality.py --lablean_dir lean --gate_only --compare_v1 results/token_quality/runs.jsonl`. Stop if it fails.
2. Then all arms: `python scripts/42_token_quality.py --lablean_dir lean --web_dir WLP-Dataset --compare_v1 results/token_quality/runs.jsonl`.

Each invocation writes a fresh output directory (default `runs/token_quality/<date-time>`) and refuses to append to an existing `runs.jsonl`. By default the number of worker processes is the free RAM (MemAvailable) divided by `GB_PER_WORKER` in the script (the peak memory of one training worker), capped at the CPU count. Check that setting against a measured peak on your machine before a full run; it was not re-measured for this revision.

## Results on eval v2

**NOT RUN.** This section will be filled in only from a real run of the commands above.

| Arm | D = 0.25M | D = 1M |
|---|---|---|
| web | not run | not run |
| agent | not run | not run |
| verified | not run | not run |
| oracle | - | not run |
| untrained | not run | |
| prior-only | 0.25 (by construction, from the data card) | |

Gate: not run.

## Old results (eval v1, first run): not valid for conclusions

These come from the first run on eval v1, with the calibrated score as primary. The section above explains why neither of its scores can be read as evidence. They are kept for the record and for the item-for-item comparison.

**Run.** `python scripts/42_token_quality.py --lablean_dir lean --web_dir WLP-Dataset` took 80 min on 7 CPU processes. Seeds: 3 per arm, 2 for the oracle. Eval v1: 482 items on 400 held-out protocols, 400 verdict items and 82 volume items. Files: [results/token_quality](../results/token_quality).

Calibrated accuracy on eval v1, mean over seeds. The untrained model scores 0.232.

| Arm | D = 0.25M | D = 1M | Raw accuracy at 1M |
|---|---|---|---|
| web | 0.254 ± 0.011 | 0.269 ± 0.014 | 0.120 |
| agent | 0.230 ± 0.090 | 0.259 ± 0.008 | 0.231 |
| verified | 0.196 ± 0.043 | 0.234 ± 0.035 | 0.241 |
| oracle | - | 0.197 ± 0.038 | 0.373 |

Untrained raw accuracy is 0.250. The question-free prior-only baseline is 0.417.

Paired differences at 1M tokens on eval v1 (95% t-interval over seeds; two-level bootstrap):

| Comparison | Difference | t-interval | Bootstrap |
|---|---|---|---|
| verified − agent | −0.025 | [−0.093, +0.043] | [−0.064, +0.013] |
| verified − web | −0.035 | [−0.155, +0.085] | [−0.095, +0.021] |
| agent − web | −0.010 | [−0.062, +0.042] | [−0.054, +0.031] |

At 0.25M every interval also includes 0.

**What the old results mean:** nothing about the arms. The oracle got 0.197 calibrated, below the untrained 0.232. Its raw 0.373 does not show that the task is learnable either, because a scorer that never reads the question gets 0.417. Both scores are dominated by the answer prior and the padding dots (see "Why the first eval cannot be used").

## Next steps

- Run the gate on eval v2, then the arms if it passes.
- Train longer or on a GPU.
- Add a record-format eval.
- Then rerun the three arms with more seeds.

## Limits

- The agent is a scripted program, and the eval is synthetic.
- Web text is far from the eval format.
- Equal characters do not carry equal information.
- 3 seeds is few; the two-level bootstrap over 3 seeds is coarse, so the t-intervals are reported too.
- Eval v2 keeps protocols that fit in the context and that fall into full blocks, so it has fewer `pass` and volume items than the natural distribution (47 passing protocols of 400, 40 volume items).
- Holding the eval out excludes exact duplicate protocols only.

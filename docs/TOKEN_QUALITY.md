# Token-quality test: web text vs agent tokens vs verified agent tokens

Question: given the same number of training tokens, do checked agent tokens teach a model more than web text?

## Setup (CPU only)

**Training arms.** Each arm gets exactly D characters, with D = 0.25M or 1M. Tokens are characters.

- **web:** real protocol text written by people, from the Wet Lab Protocols corpus (622 protocols, MIT license).
- **agent:** every record the lab token factory's agent wrote, unchecked (#97). 42.7% of these records hold a wrong claim: 36.7% a wrong verdict and 6.0% wrong volumes.
- **verified:** only records that passed LabLean, the Python copy of LabLean, and the volume replay.
- **oracle (positive control):** text in the eval's own question-and-answer format, on other protocols.
- **untrained:** the same model with no training.

**Model.** A tiny GPT with 4 layers, d=128, 4 heads, context 512 and 0.87M parameters. It trains from scratch for 400 steps at batch 8, so it sees 1.64M characters.

**Seeds.** 3 per arm (2 for the oracle).

**Eval.** 482 items: 400 "what happens?" verdict items and 82 "final volume" items. They are built on 400 held-out protocols that appear in no training arm. LabLean gives each true answer. Each item has 4 options, so chance is 25%.

**Score.** The primary score is calibrated: log p(option | question) − log p(option | "A: "). The raw score is log p(option | question).

**Run.** `python scripts/42_token_quality.py --lablean_dir lean --web_dir WLP-Dataset` took 80 min on 7 CPU processes.

## Results

Calibrated accuracy, mean over seeds. The untrained model scores 0.232.

| Arm | D = 0.25M | D = 1M | Raw accuracy at 1M |
|---|---|---|---|
| web | 0.254 ± 0.011 | 0.269 ± 0.014 | 0.120 |
| agent | 0.230 ± 0.090 | 0.259 ± 0.008 | 0.231 |
| verified | 0.196 ± 0.043 | 0.234 ± 0.035 | 0.241 |
| oracle | - | 0.197 ± 0.038 | 0.373 |

Untrained raw accuracy is 0.250.

Paired differences at 1M tokens (95% t-interval over seeds; two-level bootstrap):

| Comparison | Difference | t-interval | Bootstrap |
|---|---|---|---|
| verified − agent | −0.025 | [−0.093, +0.043] | [−0.064, +0.013] |
| verified − web | −0.035 | [−0.155, +0.085] | [−0.095, +0.021] |
| agent − web | −0.010 | [−0.062, +0.042] | [−0.054, +0.031] |

At 0.25M every interval also includes 0.

## What this means

**No conclusion about the arms yet.**

- The pre-set rule said: if the oracle does not beat the untrained model, the task is "not learnable at this scale" and nothing is concluded about the arms. On the primary (calibrated) score, the oracle got 0.197, below the untrained 0.232.
- The raw score shows the task is partly learnable when the training text matches the eval format: the oracle reached 0.373 against 0.250 untrained. Our calibration step removes that signal at this scale.
- Every paired difference has an interval that includes 0.

## Next steps

- Choose one scoring rule that the positive control passes, and fix it in advance.
- Train longer or on a GPU.
- Add a record-format eval.
- Then rerun the three arms with more seeds.

## Limits

- The agent is a scripted program, and the eval is synthetic.
- Web text is far from the eval format.
- Equal characters do not carry equal information.
- 3 seeds is few.

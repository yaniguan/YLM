# YLM v0 checkpoints

Three small checkpoints (seed 0, 60 epochs, trained on one V100 for ~35 min each). Each `.pt` holds
`state_dict`, the per-task output `scale`, and the training args.

| File | Text input | Params | Size |
|---|---|---|---|
| `ylm_matscibert_s0.pt` | frozen MatSciBERT question states (**recommended**) | 0.44M + frozen encoder | 1.8 MB |
| `ylm_minilm_s0.pt` | frozen MiniLM question states | 0.34M + frozen encoder | 1.4 MB |
| `baseline_taskid_s0.pt` | none (task ID only) | 0.25M | 1.0 MB |

The frozen text encoders are **not** in the checkpoint; `scripts/03_embed_text.py` downloads them from Hugging Face.
`results.json` has the full test and held-out numbers for these three seeds. Mean ± std over 5 seeds is in
[`../../results.md`](../../results.md).

## Use

```bash
conda activate ylm
python scripts/09_eval_checkpoint.py --ckpt checkpoints/v0/ylm_matscibert_s0.pt --demo
```

Needs `data/` from steps 1-3 of "How to run" in the main README. Re-evaluating the MatSciBERT checkpoint reproduces the
training-time numbers (force MAE 0.0068, cos 0.954; rotation error ~2e-6). `--demo` rotates one molecule and prints
that the predicted force rotates with it (difference ~1e-7).

## Intended use and limits

- Research checkpoints for TensorQA-v0 only (QM9-size organic molecules, GFN2-xTB labels, 12 question templates).
- Not a general force field or dipole predictor. No chemical-accuracy claim.
- Only seed 0 is shipped (not picked as best seed). Use `results.md` for variance across seeds.

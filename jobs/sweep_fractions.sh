#!/bin/bash
# Extra seeds for the sample-efficiency curve (3 / 10 / 30 % data; seeds 1,2 on top of seed 0).
# Matches the seed-0 protocol in runs_slurm/m1_*_f* (official v0 file, --epochs 150, --eval_rotations 32).
# Run from the repo root after `python scripts/06b_unpack_release.py --download` and `03_embed_text.py`:
#   bash jobs/sweep_fractions.sh uge     # or: slurm | local
set -euo pipefail
python -m src.provenance --check-v0 data/tensorqa_v0.json >/dev/null

sched=${1:-uge}
seeds=${SEEDS:-"1 2"}
for seed in $seeds; do
  for f in 0.03 0.1 0.3; do
    for text in minilm matscibert taskid; do
      name=frac_${text}_f${f}_s${seed}
      args=(--train_fraction "$f" --epochs 150 --eval_rotations 32 --seed "$seed" --resume)
      if [ "$sched" = slurm ]; then
        sbatch -J "$name" jobs/slurm_train.sh "$text" "$name" "${args[@]}"
      elif [ "$sched" = local ]; then
        python scripts/04_train.py --text "$text" --name "$name" "${args[@]}"
      else
        qsub -N "$name" jobs/uge_train.sh "$text" "$name" "${args[@]}"
      fi
    done
  done
done

#!/bin/bash
# Multi-seed Option A / B / B+attn_sup / C comparison on the official TensorQA-v0 file (32 rotations).
# Run from the repo root after `python scripts/06b_unpack_release.py --download` and
# `python scripts/03_embed_text.py --name minilm`:
#   bash jobs/sweep_options_v0.sh uge    # or: slurm | local
# Set SEEDS="0 1 2 3 4" for 5 seeds (default: "0 1 2").
set -euo pipefail
python -m src.provenance --check-v0 data/tensorqa_v0.json >/dev/null

sched=${1:-uge}
seeds=${SEEDS:-"0 1 2"}
text=${TEXT:-minilm}
include_b_attn_sup=${INCLUDE_B_ATTN_SUP:-1}

submit_one() {
  local name=$1
  shift
  if [ "$sched" = slurm ]; then
    sbatch -J "$name" jobs/slurm_train.sh "$text" "$name" "$@"
  elif [ "$sched" = local ]; then
    python scripts/04_train.py --text "$text" --name "$name" "$@"
  else
    qsub -N "$name" jobs/uge_train.sh "$text" "$name" "$@"
  fi
}

for seed in $seeds; do
  for opt in A B C; do
    name=v0_opt${opt}_${text}_s${seed}
    submit_one "$name" --option "$opt" --epochs 60 --eval_rotations 32 --seed "$seed" --resume
  done
  if [ "$include_b_attn_sup" = "1" ]; then
    name=v0_optB_attn0p1_${text}_s${seed}
    submit_one "$name" --option B --attn_sup 0.1 --epochs 60 --eval_rotations 32 --seed "$seed" --resume
  fi
done

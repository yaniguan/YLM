#!/bin/bash
# Extra seeds for the sample-efficiency curve (3 / 10 / 30 % data; seeds 1,2 on top of the existing seed 0).
# Run on a login node, from the repo root:  bash jobs/sweep_fractions.sh uge   (or slurm)
# For Slurm, set your allocation: SBATCH_ACCOUNT=<account> (same as sbatch -A <account>).
# 3 texts x 3 fractions x 2 seeds = 18 jobs of ~35 min on a V100.
sched=${1:-uge}
for seed in 1 2; do
  for f in 0.03 0.1 0.3; do
    for text in minilm matscibert taskid; do
      name=frac_${text}_f${f}_s${seed}
      if [ "$sched" = slurm ]; then
        sbatch -J $name jobs/slurm_train.sh $text $name --train_fraction $f --seed $seed
      else
        qsub -N $name jobs/uge_train.sh $text $name --train_fraction $f --seed $seed
      fi
    done
  done
done

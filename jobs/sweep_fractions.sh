#!/bin/bash
# Extra seeds for the sample-efficiency curve (3 / 10 / 30 % data; seeds 1,2 on top of the existing seed 0).
# Run on a login node of the cluster you use, from the repo root:  bash jobs/sweep_fractions.sh hoffman2   (or expanse)
# 3 texts x 3 fractions x 2 seeds = 18 jobs of ~35 min on a V100.
sched=${1:-hoffman2}
for seed in 1 2; do
  for f in 0.03 0.1 0.3; do
    for text in minilm matscibert taskid; do
      name=frac_${text}_f${f}_s${seed}
      if [ "$sched" = expanse ]; then
        sbatch -J $name jobs/expanse_train.sh $text $name --train_fraction $f --seed $seed
      else
        qsub -N $name jobs/hoffman2_train.sh $text $name --train_fraction $f --seed $seed
      fi
    done
  done
done

#!/bin/bash
# l=2 quadrupole question on a Hoffman2 V100.  qsub -N quad_s0 jobs/hoffman2_quad.sh quad quad_s0 --seed 0
#   arg 1: model (quad | scalar), arg 2: run name, more args go to scripts/18_train_quad.py
#$ -cwd
#$ -o runs/$JOB_NAME.$JOB_ID.out
#$ -j y
#$ -l h_rt=6:00:00,h_data=8G,gpu,V100,cuda=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ylm
python scripts/18_train_quad.py --model "$1" --name "$2" "${@:3}"

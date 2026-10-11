#!/bin/bash
# Compositional questions on a Hoffman2 V100.  qsub -N comp_bc_s0 jobs/hoffman2_comp.sh bc comp_bc_s0 --seed 0
#   arg 1: model (bc | optA), arg 2: run name, more args go to scripts/14_train_comp.py
#$ -cwd
#$ -o runs/$JOB_NAME.$JOB_ID.out
#$ -j y
#$ -l h_rt=12:00:00,h_data=8G,gpu,V100,cuda=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ylm
python scripts/14_train_comp.py --model "$1" --name "$2" "${@:3}"

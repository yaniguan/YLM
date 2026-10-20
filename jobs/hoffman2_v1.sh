#!/bin/bash
# TensorQA-v1 on a Hoffman2 V100.  qsub -N v1_bc_s0 jobs/hoffman2_v1.sh --model bc --name v1_bc_s0 --seed 0
#   all args go to scripts/22_train_v1.py (the text encoder must already be in the HF cache: compute nodes are offline)
#$ -cwd
#$ -o runs/$JOB_NAME.$JOB_ID.out
#$ -j y
#$ -l h_rt=14:00:00,h_data=12G,gpu,V100,cuda=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ylm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/22_train_v1.py "$@"

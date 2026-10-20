#!/bin/bash
# TensorQA-pol on a Hoffman2 V100.  qsub -N pol_s0 jobs/hoffman2_pol.sh --name pol_s0 --seed 0
#$ -cwd
#$ -o runs/$JOB_NAME.$JOB_ID.out
#$ -j y
#$ -l h_rt=10:00:00,h_data=12G,gpu,V100,cuda=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ylm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/25_train_pol.py "$@"

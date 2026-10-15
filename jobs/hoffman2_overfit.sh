#!/bin/bash
# LLM baseline sanity check on a V100: qsub -N of jobs/hoffman2_overfit.sh
#$ -cwd
#$ -o runs/$JOB_NAME.$JOB_ID.out
#$ -j y
#$ -l h_rt=1:30:00,h_data=16G,gpu,V100,cuda=1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ylm
export HF_HUB_OFFLINE=1
for cfg in "2e-5 1e-3" "1e-5 1e-3" "5e-6 1e-3" "2e-5 1e-4"; do python scripts/05b_overfit_check.py $cfg 150 2>&1 | grep "^lr"; done

#!/bin/bash
# Train one model on a Hoffman2 V100. (This account can only get V100 / generic gpu; A100, H100, L40S, H200 are refused.)
# Submit from $SCRATCH/YLM:   qsub -N ylm_minilm jobs/hoffman2_train.sh minilm ylm_minilm
# Other GPU type: add  -l h_rt=8:00:00,h_data=6G,gpu,RTX2080Ti,cuda=1  to qsub (overrides the line below)
#   arg 1: text model (minilm | matscibert | taskid)
#   arg 2: run name (results go to runs/<name>)
#   more args are passed to scripts/04_train.py
#$ -cwd
#$ -o runs/$JOB_NAME.$JOB_ID.out
#$ -j y
#$ -l h_rt=8:00:00,h_data=6G,gpu,V100,cuda=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ylm
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/04_train.py --text "$1" --name "$2" "${@:3}"

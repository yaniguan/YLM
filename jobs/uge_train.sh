#!/bin/bash
# Train one model on one UGE GPU (V100 by default).
# Submit from the repo root:   qsub -N ylm_minilm jobs/uge_train.sh minilm ylm_minilm
# Other GPU type: pass your own  -l ...  resource line to qsub (overrides the line below)
#   arg 1: text model (minilm | matscibert | taskid)
#   arg 2: run name (results go to runs/<name>)
#   more args are passed to scripts/04_train.py
#$ -cwd
#$ -o runs/$JOB_NAME.$JOB_ID.out
#$ -j y
#$ -l h_rt=8:00:00,h_data=6G,gpu,V100,cuda=1

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${YLM_ENV:-$HOME/envs/ylm}"
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/04_train.py --text "$1" --name "$2" "${@:3}"

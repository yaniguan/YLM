#!/bin/bash
# Baseline (ii) on one UGE GPU: coordinates as text -> Qwen2.5-0.5B (float32, batch 8, grad checkpointing).
# Submit from the repo root:   qsub -N llm05_aug jobs/uge_llm.sh llm05_aug --augment
#   arg 1: run name, more args go to scripts/05_train_llm_baseline.py
# Download the model once on a node with internet first (compute nodes may have no internet).
#$ -cwd
#$ -o runs/$JOB_NAME.$JOB_ID.out
#$ -j y
#$ -l h_rt=24:00:00,h_data=16G,gpu,V100,cuda=1

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${YLM_ENV:-$HOME/envs/ylm}"
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/05_train_llm_baseline.py --llm Qwen/Qwen2.5-0.5B --name "$1" "${@:2}"

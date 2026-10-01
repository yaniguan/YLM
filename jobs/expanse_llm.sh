#!/bin/bash
# Baseline (ii) on an Expanse V100: coordinates as text -> Qwen2.5-0.5B.
# Submit from ~/YLM:   sbatch -J llm05_aug jobs/expanse_llm.sh llm05_aug --augment
#   arg 1: run name, more args go to scripts/05_train_llm_baseline.py
#SBATCH --account=cla175
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x.%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ylm
export HF_HOME=~/hf HF_HUB_OFFLINE=1   # model is downloaded on the login node first
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/05_train_llm_baseline.py --llm Qwen/Qwen2.5-0.5B --name "$1" "${@:2}"

#!/bin/bash
# Baseline (ii) on one Slurm GPU: coordinates as text -> Qwen2.5-0.5B.
# Submit from the repo root:   sbatch -A <account> -J llm05_aug jobs/slurm_llm.sh llm05_aug --augment
#   Pass your allocation with `sbatch -A <account>` and a GPU partition with `-p <partition>` if needed.
#   arg 1: run name, more args go to scripts/05_train_llm_baseline.py
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x.%j.out

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${YLM_ENV:-$HOME/envs/ylm}"
export HF_HOME="${HF_HOME:-$HOME/hf}" HF_HUB_OFFLINE=1   # download the model once on a node with internet first
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/05_train_llm_baseline.py --llm Qwen/Qwen2.5-0.5B --name "$1" "${@:2}"

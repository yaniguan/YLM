#!/bin/bash
# Train one model on one Slurm GPU.
# Submit from the repo root:   sbatch -A <account> -J ylm_minilm jobs/slurm_train.sh minilm ylm_minilm
#   Pass your allocation with `sbatch -A <account>` and a GPU partition with `-p <partition>` if needed.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=runs/%x.%j.out

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${YLM_ENV:-$HOME/envs/ylm}"
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/04_train.py --text "$1" --name "$2" "${@:3}"

#!/bin/bash
# Train one model on an Expanse GPU (V100).
# Submit from ~/YLM:   sbatch -J ylm_minilm jobs/expanse_train.sh minilm ylm_minilm
#SBATCH --account=cla175
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=runs/%x.%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ylm
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/04_train.py --text "$1" --name "$2" "${@:3}"

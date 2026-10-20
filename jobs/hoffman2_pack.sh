#!/bin/bash
# Run several small trainings concurrently on ONE GPU (the account is capped at 4 GPUs at a time).
#   qsub -N pack_a0 jobs/hoffman2_pack.sh jobs/packs/a_s0.txt
# Each line of the pack file is a full command (include "> runs/<name>.stdout 2>&1"). All lines start together.
#$ -cwd
#$ -o runs/$JOB_NAME.$JOB_ID.out
#$ -j y
#$ -pe shared 5
#$ -l h_rt=24:00:00,h_data=6G,gpu,cuda=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ylm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  bash -c "$cmd" &
  sleep 20      # stagger the starts so peak memory does not coincide
done < "$1"
wait

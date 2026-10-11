#!/bin/bash
# l=2 quadrupole labels (PySCF HF/6-31G*, ~4 s per molecule): 2000 molecules = 10 tasks x 200.
#   qsub -t 1-10 jobs/quadrupole_array.sh   ->  data/quad_part<k>.pkl ; merge with scripts/12_make_quadrupole.py output format (dict mol -> 3x3)
#$ -cwd
#$ -o runs/quad.$JOB_ID.$TASK_ID.out
#$ -j y
#$ -l h_rt=2:00:00,h_data=4G
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ylm
k=$((SGE_TASK_ID-1))
python scripts/12_make_quadrupole.py --start $((k*200)) --stop $(((k+1)*200)) --out data/quad_part$SGE_TASK_ID.pkl

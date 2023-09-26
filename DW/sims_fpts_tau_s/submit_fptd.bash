#!/bin/bash
#SBATCH -p compute # partition 
#SBATCH -N 1 # number of nodes 
#SBATCH -n 50 # number of cores 
#SBATCH --mem 128G # memory pool for all cores 
#SBATCH -t 0-01:00 # time (D-HH:MM) 
#SBATCH --output=out/out_%a.out
#SBATCH --error=out/err_%a.out

module load python/3.7.3

python3 -u compute_fptd.py -eps_idx ${SLURM_ARRAY_TASK_ID} 



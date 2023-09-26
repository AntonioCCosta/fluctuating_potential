#!/bin/bash
#SBATCH -p compute # partition 
#SBATCH -N 1 # number of nodes 
#SBATCH -n 50 # number of cores 
#SBATCH --mem 32G # memory pool for all cores 
#SBATCH -t 1-00:00 # time (D-HH:MM) 
#SBATCH --output=output_path/fptds_%a.out
#SBATCH --error=out/err_%a.out

module load python/3.7.3

python3 -u sims_fpt.py -idx ${SLURM_ARRAY_TASK_ID}



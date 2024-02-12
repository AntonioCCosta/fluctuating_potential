#!/bin/bash
#SBATCH -p compute # partition 
#SBATCH -N 1 # number of nodes 
#SBATCH -n 50 # number of cores 
#SBATCH --mem 32G # memory pool for all cores 
#SBATCH -t 0-08:00 # time (D-HH:MM) 
#SBATCH --output=out_%A.out
#SBATCH --error=err_%A.out

module load python/3.7.3

python3 -u estimate_fpt_errbars.py


#!/bin/bash
#SBATCH -p compute # partition 
#SBATCH -N 1 # number of nodes 
#SBATCH -n 12 # number of cores 
#SBATCH --mem 32G # memory pool for all cores 
#SBATCH -t 0-01:00 # time (D-HH:MM) 
#SBATCH --output=out/out_%A_%a.out                                         
#SBATCH --error=out/err_%A_%a.out

module load python/3.7.3

python3 simulate_symbol_seqs.py



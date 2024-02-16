#!/bin/bash
#
#SBATCH --job-name=perpt
#SBATCH --time=02:00:00
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=agu1@le.ac.uk
#SBATCH --export=NONE

python3 permuted_100_hpc.py

#!/bin/bash
#
#SBATCH --job-name=objd
#SBATCH --time=02:00:00
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mpk20@le.ac.uk
#SBATCH --export=NONE

source ~/miniconda3/bin/activate medcondaenv
python3 R-CNN-Object-Detection-Copy2.py

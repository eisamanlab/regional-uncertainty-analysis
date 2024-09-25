#!/bin/bash

#SBATCH --job-name=analysis
#SBATCH --ntasks=1
#SBATCH --time=0:10:00
#SBATCH --account eisaman
#SBATCH --nodes 1
#SBATCH --mem 20G
#SBATCH --partition day

module purge
module load miniconda
conda activate dev

python3 ./analysis.py
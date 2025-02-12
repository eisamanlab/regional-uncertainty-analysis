#!/bin/bash

#SBATCH --job-name=analysis
#SBATCH --ntasks=1
#SBATCH --time=0:20:00
#SBATCH --account eisaman
#SBATCH --nodes 1
#SBATCH --mem 25G
#SBATCH --partition day

module purge
module load miniconda
conda activate dev

python3 ./analysis.py
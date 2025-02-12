#!/bin/bash

#SBATCH --job-name=oras5
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --account eisaman
#SBATCH --nodes 1
#SBATCH --mem 10G
#SBATCH --partition day

module purge
module load miniconda
conda activate dev

# ----------------------------------------
# 1. make list of urls to download
# completese in seconds
# ----------------------------------------
python download-data-1993-2014.py
wait

python download-data-2015-2022.py

wait



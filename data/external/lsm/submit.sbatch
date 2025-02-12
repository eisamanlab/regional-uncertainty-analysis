#!/bin/bash

#SBATCH --job-name=create-mask
#SBATCH --ntasks=1
#SBATCH --time=5:00
#SBATCH --account eisaman
#SBATCH --nodes 1
#SBATCH --mem 5G
#SBATCH --partition day

module purge
module load miniconda
conda activate dev

python3 /home/ljg48/project/oae-uncertainty/data/external/lsm/create_ocean_mask.py 


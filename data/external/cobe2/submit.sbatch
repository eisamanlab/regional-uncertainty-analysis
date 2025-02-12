#!/bin/bash

#SBATCH --job-name=cobe2
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --account eisaman
#SBATCH --nodes 1
#SBATCH --mem 10G
#SBATCH --partition day

module purge
module load miniconda
conda activate dev

# -----------------------------------------
# process data
# -----------------------------------------
python3 ./process_cobe2.py

wait


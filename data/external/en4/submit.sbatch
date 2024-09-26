#!/bin/bash

#SBATCH --job-name=en4_pr
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --account eisaman
#SBATCH --nodes 1
#SBATCH --mem 5G
#SBATCH --partition day

module purge
module load miniconda
conda activate dev

start_year=1982
end_year=2023
input_file='/home/ljg48/palmer_scratch/data/EN.4.2.2.analyses/scripts/EN.4.2.2.analyses.c13.download-list.txt'
raw_directory='/home/ljg48/palmer_scratch/data/EN.4.2.2.analyses/raw'
processed_directory='/home/ljg48/palmer_scratch/data/EN.4.2.2.analyses/processed'

# ----------------------------------------
# 1. download files
# ----------------------------------------
#./download_files.sh $input_file $raw_directory $start_year $end_year

# ----------------------------------------
# 2. process zip files
# ---------------------------------------
python ./process_zip_files.py --input ${raw_directory} --output ${processed_directory}


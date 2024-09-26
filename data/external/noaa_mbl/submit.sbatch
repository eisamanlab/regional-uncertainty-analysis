#!/bin/bash

#SBATCH --job-name=noaa-mbl-pr
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --account eisaman
#SBATCH --nodes 1
#SBATCH --mem 1G
#SBATCH --partition day

module purge
module load miniconda
conda activate dev

url="https://gml.noaa.gov/ccgg/mbl/tmp/co2_GHGreference.169228596_surface.txt"
#raw_dir="/home/ljg48/palmer_scratch/data/noaa-mbl/raw"
raw_dir="/home/ljg48/project/data/noaa-mbl/raw"
file_name="${url##*/}"
input_file="${raw_dir}/${file_name}"
output_file="/home/ljg48/palmer_scratch/data/noaa-mbl/processed/noaa-mbl_197901-202301_1x1.nc"

# ----------------------------------------
# 1. download raw data
# ----------------------------------------
wget ${url} -P ${raw_dir}

# ----------------------------------------
# 2. process noaa mbl data 
# ----------------------------------------
#python process-noaa-mbl.py --input ${input_file} --output ${output_file}


#!/bin/bash

#SBATCH --job-name=gcb23-pr
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --account eisaman
#SBATCH --nodes 1
#SBATCH --mem 20G
#SBATCH --partition day

module purge
module load miniconda
conda activate dev

models="./gcb2023-ocean-models.txt"
dataprods="./gcb2023-data-products.txt"
models_dir="/home/ljg48/palmer_scratch/data/pCO2/raw/models"
dataprods_dir="/home/ljg48/palmer_scratch/data/pCO2/raw/data-products"

start_year=1990
end_year=2022
models_ofile="/home/ljg48/palmer_scratch/data/pCO2/processed/gcb-2023-models_1x1.nc"
dataprods_ofile="/home/ljg48/palmer_scratch/data/pCO2/processed/gcb-2023-dataprods_1x1.nc"

# ----------------------------------------
# 1. download data 
# ----------------------------------------
#wget -i ${models} -P ${models_dir}
#wget -i ${dataprods} -P ${dataprods_dir}

# ----------------------------------------
# 2.1 process models
# ----------------------------------------
#python process-models.py --input ${models_dir} --output ${models_ofile} --start-year ${start_year} --end-year ${end_year}

# -----------------------------------------
# 2.2. process data products
# -----------------------------------------
python process-dataprods.py --input ${dataprods_dir} --output ${dataprods_ofile} --start-year ${start_year} --end-year ${end_year}

wait

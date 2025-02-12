#!/bin/bash

#SBATCH --job-name=esa
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --account eisaman
#SBATCH --nodes 1
#SBATCH --mem 10G
#SBATCH --partition day

module purge
module load miniconda
conda activate dev

batch_size=5
url_list='/home/ljg48/palmer_scratch/esa/url-list-corrupt-file.txt'
raw_directory='/home/ljg48/palmer_scratch/esa/raw'
processed_directory='/home/ljg48/palmer_scratch/esa/processed'

# ----------------------------------------
# 1. make list of urls to download
# completese in seconds
# run this and then run wget
# ----------------------------------------
python create-url-list.py 

wait
# ----------------------------------------
# 2. download data from url_list
# ----------------------------------------
wget -i ${url_list} -P "${raw_directory}"

wait 
# ----------------------------------------
# 3. process data: look at README 
# for instructions to use dsq 
# ----------------------------------------


#!/bin/bash

#SBATCH --job-name=ostia
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --account eisaman
#SBATCH --nodes 1
#SBATCH --mem 10G
#SBATCH --partition day

source /home/ljg48/.nasa-earthdata

module purge
module load miniconda
conda activate dev

batch_size=5
url_list='/home/ljg48/palmer_scratch/ostia/url-list.txt'
raw_directory='/home/ljg48/palmer_scratch/ostia/raw'
processed_directory='/home/ljg48/palmer_scratch/ostia/processed'
USER=${NASA_USER}
PASSWORD=${NASA_PASSWORD}

# ----------------------------------------
# 1. make list of urls to download
# completese in seconds
# ----------------------------------------
python create-url-list.py 

wait

# ----------------------------------------
# 2. download data from url_list
# ----------------------------------------
wget --user ${USER} --password ${PASSWORD} -i ${url_list} -P "${raw_directory}"



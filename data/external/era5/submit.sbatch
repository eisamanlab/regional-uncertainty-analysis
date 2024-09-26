#!/bin/bash

#SBATCH --job-name=era5_pr
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --account eisaman
#SBATCH --nodes 1
#SBATCH --mem 10G
#SBATCH --partition day

module purge
module load miniconda
conda activate dev

start_year=1982
end_year=2023
batch_size=5
url_list='/home/ljg48/palmer_scratch/data/ERA5/scripts/url-list.txt'
raw_directory='/home/ljg48/palmer_scratch/data/ERA5/raw'
processed_directory='/home/ljg48/palmer_scratch/data/ERA5/processed'

# ----------------------------------------
# 1. make list of urls to download
# completese in seconds
# ----------------------------------------
#python create-url-list.py --start-year ${start_year} --end-year ${end_year} --output-file ${url_list} 

# ----------------------------------------
# 2. download data from url_list
# ----------------------------------------
# wget -i ${url_list} -P "${output_directory}"

# ----------------------------------------
# 3. process data
# ----------------------------------------
for (( year=$start_year; year<=$end_year; year+=$batch_size ))
do
    for (( batch_year=year; batch_year<year+$batch_size && batch_year<=$end_year; batch_year++ ))
    do
        python3 ./process_year.py --year ${batch_year} --input ${raw_directory} --output ${processed_directory} &
    done

    wait
done

wait


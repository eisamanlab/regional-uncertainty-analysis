#!/bin/bash

#SBATCH --job-name=ccmp_pr
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --account eisaman
#SBATCH --nodes 1
#SBATCH --mem 20G
#SBATCH --partition day

module purge
module load miniconda
conda activate dev

start_year=1993
end_year=1996
batch_size=1
url_list='/home/ljg48/palmer_scratch/data/CCMP/scripts/url-list-ccmp-v3.1_4.txt'
raw_directory='/home/ljg48/palmer_scratch/data/CCMP/raw'
processed_directory='/home/ljg48/palmer_scratch/data/CCMP/processed'

# ----------------------------------------
# 1. make list of urls to download
# completese in seconds
# ----------------------------------------
#python create-url-list-ccmpv3.1.py --start-year ${start_year} --end-year ${end_year} --output-file ${url_list} 

# ----------------------------------------
# 2. download the data
# ----------------------------------------
#wget -i ${url_list} -P ${raw_directory}

# ----------------------------------------
# 3. process year
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

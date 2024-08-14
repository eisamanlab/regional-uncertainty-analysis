#!/bin/bash

#SBATCH --job-name=process_oisst_2000
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --account eisaman
#SBATCH --nodes 1
#SBATCH --mem 20G
#SBATCH --partition day

module purge
module load miniconda
conda activate dev

# ----------------------------------------
# 1. get the links (takes ~5 minutes)
# commend this out if not running it
# ----------------------------------------
#python /home/ljg48/palmer_scratch/data/OISST/scripts/get_oisst_links.py

# ----------------------------------------
# 2. download files (takes ~4 hours)
#    using batch size of 100
# command this out if not running it
# ----------------------------------------
#./download_files.sh -d ../raw/ -f ./oisst_links.txt

# -----------------------------------------
# 3. process each year (takes ~1 hour)
#    does not seem to run in parallel
# -----------------------------------------
start_year=2000
end_year=2000
batch_size=5

for (( year=$start_year; year<=$end_year; year+=$batch_size ))
do
    for (( batch_year=year; batch_year<year+$batch_size && batch_year<=$end_year; batch_year++ ))
    do
        python3 ./process_year.py --year ${batch_year} &
    done

    wait
done


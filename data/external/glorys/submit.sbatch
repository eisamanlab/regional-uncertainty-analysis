#!/bin/bash

#SBATCH --job-name=glorys
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --account eisaman
#SBATCH --nodes 1
#SBATCH --mem 25G
#SBATCH --partition day

module purge
module load miniconda
conda activate dev

# ----------------------------------------
# 1. download data
# ----------------------------------------
python download_data.py

wait 

python download-data-2022.py

wait

# -----------------------------------------
# 3. process each year (takes ~1 hour)
#    does not seem to run in parallel
# -----------------------------------------
start_year=1993
end_year=2022

for year in $(seq $start_year $end_year); do
    python3 ./process_year.py --year ${year} 
done

wait


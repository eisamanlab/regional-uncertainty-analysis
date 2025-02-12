#!/bin/bash
#SBATCH --output dsq-joblist-%A_%2a-%N.out
#SBATCH --array 0-29
#SBATCH --job-name esa-dsq
#SBATCH --mem-per-cpu 25g -t 2:00:00

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/gibbs/project/eisaman/ljg48/oae-uncertainty/data/external/esa_sst_cci/joblist.txt --status-dir /gpfs/gibbs/project/eisaman/ljg48/oae-uncertainty/data/external/esa_sst_cci


#!/bin/bash
#SBATCH --output dsq-joblist-%A_%2a-%N.out
#SBATCH --array 0-29
#SBATCH --job-name oras5-dsq
#SBATCH --mem-per-cpu 25g -t 1:00:00

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/gibbs/project/eisaman/ljg48/oae-uncertainty/data/external/oras5/joblist.txt --status-dir /gpfs/gibbs/project/eisaman/ljg48/oae-uncertainty/data/external/oras5


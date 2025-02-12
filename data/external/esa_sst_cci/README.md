# ESA

## Source 
https://catalogue.ceda.ac.uk/uuid/4a9654136a7148e39b7feb56f8bb02d2/

## Notes
For more information, check out: https://climate.esa.int/en/projects/sea-surface-temperature/

## Steps
1. Download the data. Run the commands in `submit.sbatch`, may need to change directory names

2. generates joblist using following command

```bash
seq 1993 2022 | xargs -I {} echo "module load miniconda; conda activate dev; python3 ./process_year.py --year {}" > joblist.txt
```

3. run `module load dSQ` on Yale's Grace cluster

4. create submit script

```
dsq --job-file joblist.txt --mem-per-cpu 25g -t 2:00:00 --job-name esa-dsq --batch-file dsq_submit.sbatch
```

5. submit script `sbatch dsq_submit.sbatch`. Alternatively, if you are not on Yale's cluster, you can run each command separetly in the joblist.


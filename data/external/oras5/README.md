# ORAS5

## Source
https://cds.climate.copernicus.eu/datasets/reanalysis-oras5?tab=overview

## Citation

[Zuo et al. (2019)](https://doi.org/10.5194/os-15-779-2019)

## Notes
0. Download the data, use `submit.sbatch`

1. generates joblist

```bash
seq 1993 2022 | xargs -I {} echo "module load miniconda; conda activate dev; python3 ./process_year.py --year {}" > joblist.txt
```

2. in Yale's Grace cluster run: `module load dSQ`

3. create submit script

```bash
dsq --job-file joblist.txt --mem-per-cpu 25g -t 2:00:00 --job-name esa-dsq --batch-file dsq_submit.sbatch
```

4. submit script `dsq_submit.sbatch`. Alternatively you can run each job in the joblist separately

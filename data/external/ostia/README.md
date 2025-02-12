# OSTIA

You will need a [NASA EarthData account](https://urs.earthdata.nasa.gov/) to download this data

## Source
https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_REP_OBSERVATIONS_010_011/files?subdataset=METOFFICE-GLO-SST-L4-REP-OBS-SST_202003

## Citation

[Good et al. (2020)](https://doi.org/10.3390/rs12040720)

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

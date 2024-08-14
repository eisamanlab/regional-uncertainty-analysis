""" 
process a single year of oisst data
>> python process_year.py --year <year_to_process> --input <path_to_input_data> --output <path_where_data_saved>
author: L. Gloege
created: 2024-05-17
"""

import argparse
from pathlib import Path
from time import time 

import numpy as np
import xarray as xr
import xesmf as xe


def time_this_function(func):
    """timer decorator"""
    def wrapper(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time() 
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s with args: {args} and kwargs: {kwargs}') 
        return result
    return wrapper

@time_this_function
def main(year: int, input: str, output: str):
    output_path = Path(output)

    ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-89.5, 90, 1.0), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(0.5, 360, 1.0), {"units": "degrees_east"}),
        }
    )
    
    # load data for specific year
    path = Path(input)
    files = path.glob(f'*.{year}????.nc')
    ds = xr.open_mfdataset([file for file in files])

    # removes zlev
    ds = ds.squeeze()
    
    # average monthly
    ds_avg = ds.resample(time="1MS").mean()
    
    # new time vector centered on 15th month
    ds_tmp = ds_avg.assign_coords(time = ds_avg.time + np.timedelta64(14, 'D'))
    
    regridder = xe.Regridder(ds_tmp, ds_out, method="bilinear")
    ds_regrid = regridder(ds_tmp, keep_attrs=True)
    ds_regrid = ds_regrid.reset_coords()
    
    # save to file
    dataset_name = next(path.glob(f'*.{year}????.nc')).stem.split('.')[0]
    output_file = f"{dataset_name}.1x1.{year}.nc"
    ds_regrid.compute().to_netcdf(output_path / output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        help="year to process",
        default=1982,
    )
    parser.add_argument(
        "--output",
        help="path to save files to",
        default="/home/ljg48/palmer_scratch/data/OISST/processed/",
    )
    parser.add_argument(
        "--input",
        help="path to raw data",
        default="/home/ljg48/palmer_scratch/data/OISST/raw",
    )
    args = parser.parse_args()

    year = args.year
    input = args.input
    output = args.output
    
    main(year=year, input=input, output=output)
""" 
process cobe2 sst data
>> python process_cobe2.py --input <path_to_input_data> --output <path_where_data_saved>
author: L. Gloege
created: 2025-01-29
"""

import argparse
from pathlib import Path
from time import time 

import numpy as np
import xarray as xr


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
def main(input: str, output: str):
    output_path = Path(output)
    
    input_file = f"{input}/sst.mon.mean.nc"
    
    start_year = 1993
    end_year = 2022

    # processing
    ds = (
        xr.open_dataset(input_file)
        .sel(time=slice(str(start_year), str(end_year)))
        .sortby("lat")
        .assign_coords(time = lambda ds: ds.time + np.timedelta64(14, 'D'))
    )

    # save output
    dataset_name = "cobe2"
    output_file = f"{dataset_name}.1x1.{start_year}-{end_year}.nc"
    ds.to_netcdf(output_path / output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        help="path to save files to",
        default="/home/ljg48/palmer_scratch/cobe2/processed/",
    )
    parser.add_argument(
        "--input",
        help="path to raw data",
        default="/home/ljg48/palmer_scratch/cobe2/raw",
    )
    args = parser.parse_args()

    input = args.input
    output = args.output
    
    main(input=input, output=output)

""" 
process a single year of esa data
>> python process_year.py --year <year_to_process> --input <path_to_input_data> --output <path_where_data_saved>
author: L. Gloege
created: 2025-01-28
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
            "lon": (["lon"], np.arange(-179.5, 180, 1.0), {"units": "degrees_east"}),
        }
    )
    
    # load data for specific year
    path = Path(input)
    files = path.glob(f"*{year}????120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_*.nc")
    ds = xr.open_mfdataset([file for file in files])[['analysed_sst', 'analysed_sst_uncertainty', 'sea_ice_fraction']]

    # change longitude to go from 0 - 360
    ds = ds.assign(lon = [l if l>0 else l+360 for l in ds.lon])
    ds = ds.sortby("lon")

    # average monthly
    ds_avg = ds.resample(time="1MS").mean()
    
    # new time vector centered on 15th month
    ds_tmp = ds_avg.assign_coords(time = ds_avg.time + np.timedelta64(14, 'D'))
    
    regridder = xe.Regridder(ds_tmp, ds_out, method="bilinear")
    ds_regrid = regridder(ds_tmp, keep_attrs=True)
    ds_regrid = ds_regrid.reset_coords()

    # change longitude to go from 0 - 360
    ds_regrid = (
        ds_regrid.assign_coords(lon = [l if l>0 else l+360 for l in ds_regrid.lon])
        .sortby("lon")
    )
    
    # save to file
    dataset_name = "esa-cci"
    output_file = f"{dataset_name}.1x1.{year}.nc"
    ds_regrid.compute().to_netcdf(output_path / output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        help="year to process",
        default=1993,
    )
    parser.add_argument(
        "--output",
        help="path to save files to",
        default="/home/ljg48/palmer_scratch/esa/processed/",
    )
    parser.add_argument(
        "--input",
        help="path to raw data",
        default="/home/ljg48/palmer_scratch/esa/raw",
    )
    args = parser.parse_args()

    year = args.year
    input = args.input
    output = args.output
    
    main(year=year, input=input, output=output)

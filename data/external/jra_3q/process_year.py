"""
process a single year of JRA3Q data
>> python process_year.py --year <year_to_process> --input <path_to_input_data> --output <path_where_data_saved>
author: L. Gloege
created: 2024-05-28
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
        print(
            f"Function {func.__name__!r} executed in {(t2-t1):.4f}s with args: {args} and kwargs: {kwargs}"
        )
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
    files = path.glob(
        f"jra3q.anl_surf.0_2_?.?grd10m-hgt-an-gauss.{year}??????_{year}??????.nc"
    )
    ds = xr.open_mfdataset(files)

    # calculate wind speed
    ds = ds.assign(
            ws = lambda ds: (ds["ugrd10m-hgt-an-gauss"]**2 + ds["vgrd10m-hgt-an-gauss"]**2)**(0.5)
        )

    # average monthly
    ds_avg_tmp = ds.resample(time="1MS").mean()
    ds_std_tmp = ds.resample(time="1MS").std()

    # new time vector centered on 15th month
    ds_avg = (
        ds_avg_tmp[["ws", "ugrd10m-hgt-an-gauss", "vgrd10m-hgt-an-gauss"]]
        .rename(
            {
                "ws": "ws_mean",
                "ugrd10m-hgt-an-gauss": "uwnd_mean",
                "vgrd10m-hgt-an-gauss": "vwnd_mean",
            }
        )
        .assign_coords(time=ds_avg_tmp.time + np.timedelta64(14, "D"))
    )

    ds_std = (
        ds_std_tmp[["ws", "ugrd10m-hgt-an-gauss", "vgrd10m-hgt-an-gauss"]]
        .rename(
            {
                "ws": "ws_std",
                "ugrd10m-hgt-an-gauss": "uwnd_std",
                "vgrd10m-hgt-an-gauss": "vwnd_std",
            }
        )
        .assign_coords(time=ds_std_tmp.time + np.timedelta64(14, "D"))
    )

    ds_tmp = xr.merge([ds_avg, ds_std])

    regridder = xe.Regridder(ds_tmp, ds_out, method="bilinear")
    ds_regrid = regridder(ds_tmp, keep_attrs=True)
    ds_regrid = ds_regrid.reset_coords()

    # save to file
    output_file = f"jra3q.anl_surf.1x1.{year}.nc"
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
        default="/home/ljg48/palmer_scratch/data/JRA3Q/processed/",
    )
    parser.add_argument(
        "--input",
        help="path to raw data",
        default="/home/ljg48/palmer_scratch/data/JRA3Q/raw",
    )
    args = parser.parse_args()

    year = args.year
    input = args.input
    output = args.output

    main(year=year, input=input, output=output)

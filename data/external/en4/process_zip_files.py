import argparse
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe


def print_kwargs(func):
    """decorator to display input and output"""

    def wrapper(*args, **kwargs):
        input = kwargs.get("input", 'No "input" argument provided')
        output = kwargs.get("output", 'No "output" argument provided')
        result = func(*args, **kwargs)
        print(f"{input} --> {output}")
        return result

    return wrapper


@print_kwargs
def process_zip_file(zip_file: Path, output_dir: Path):
    ds_grid = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-89.5, 90, 1.0), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(0.5, 360, 1.0), {"units": "degrees_east"}),
        }
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # add following to chunk: .chunk(chunks={'time': 'auto', 'lat': 100, 'lon': 100})
        ds = (
            xr.open_mfdataset(Path(temp_dir).glob("*.nc"))
            .pipe(lambda ds: ds.sel(depth=ds.depth.min().values))
            .sortby(["lat", "lon"])
        )

    # regrid data
    regridder = xe.Regridder(ds, ds_grid, method="bilinear")
    ds_regrid = regridder(ds, keep_attrs=True)

    # create time vector centered on 15th of month
    years = ds_regrid.time.dt.year.values
    months = ds_regrid.time.dt.month.values
    days = [15] * len(years)

    timestamps = pd.to_datetime({"year": years, "month": months, "day": days})
    ds_out = ds_regrid.assign_coords({"time": timestamps})
    ds_out = ds_out.compute()

    # save as netcdf to output_file
    output_file = output_dir / f"{zip_file.stem}.monthly.1x1.nc"
    ds_out.to_netcdf(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="directory wih zip files",
        default="/home/ljg48/palmer_scratch/data/EN.4.2.2.analyses/tmp",
    )
    parser.add_argument(
        "--output",
        help="directory to save processed netcdf files to",
        default="/home/ljg48/palmer_scratch/data/EN.4.2.2.analyses/raw",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    for zip_file in input_dir.glob("*.zip"):
        process_zip_file(zip_file=zip_file, output_dir=output_dir)

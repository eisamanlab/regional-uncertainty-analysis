import argparse
from datetime import datetime, timedelta
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def deg_to_rad(deg):
    return deg * (math.pi / 180)


def rad_to_deg(rad):
    return rad * (180 / math.pi)


def decimal_year_to_datetime(dec):
    year = int(dec)
    rem = dec - year
    base = datetime(year, 1, 1)
    return base + timedelta(
        seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem
    )


def main(input: Path, output: Path):
    with open(input, "r") as file:
        for line in file:
            if "Sine of latitude steps:" in line:
                sine_of_lat_strings = line.split(":")[1].strip().split()
                sine_of_lat = [float(val) for val in sine_of_lat_strings]
                lats = [rad_to_deg(math.asin(num)) for num in sine_of_lat]
                break  # Stop reading after finding the desired line

    column_names = ["decimal_year"] + [str(lat) for lat in lats]

    df = pd.read_csv(
        input, sep="\s+", header=None, comment="#", skip_blank_lines=True
    )  # names = column_names)

    df = df.set_index(0)

    xco2 = df.iloc[:, ::2].reset_index()
    xco2 = xco2.rename(columns=dict(zip(xco2.columns, column_names)))

    xco2_uncert = df.iloc[:, 1::2].reset_index()
    xco2_uncert = xco2_uncert.rename(
        columns=dict(zip(xco2_uncert.columns, column_names))
    )

    # datetimes = [decimal_year_to_datetime(val) for val in xco2['decimal_year']]
    # xco2_uncert['datetime'] = datetimes

    df_uncert = xco2_uncert.melt(
        id_vars=["decimal_year"], var_name="latitude", value_name="xco2_uncertainty"
    )
    # Convert latitude column to float
    df_uncert["latitude"] = df_uncert["latitude"].astype(float)
    df_uncert = df_uncert.sort_values(by="decimal_year").reset_index(drop=True)

    df_xco2 = xco2.melt(
        id_vars=["decimal_year"], var_name="latitude", value_name="xco2"
    )
    # Convert latitude column to float
    df_xco2["latitude"] = df_xco2["latitude"].astype(float)
    df_xco2 = df_xco2.sort_values(by="decimal_year").reset_index(drop=True)

    df_out = pd.merge(left=df_xco2, right=df_uncert, on=["decimal_year", "latitude"])

    datetimes = [decimal_year_to_datetime(val) for val in df_out["decimal_year"]]
    df_out["datetime"] = datetimes

    # Then average monthly

    # Extract year from datetime
    df_out["year"] = df_out["datetime"].dt.year
    df_out["month"] = df_out["datetime"].dt.month

    df_avg = (
        df_out.groupby(by=["year", "month", "latitude"])
        .mean(["xco2", "xoc2_uncertainty"])
        .reset_index()
    )

    # Create a datetime column by combining year, month, and setting the day to 15
    df_avg["datetime"] = pd.to_datetime(
        df_avg["year"].astype(str) + "-" + df_avg["month"].astype(str) + "-15T00:00:00"
    )

    # conver tto xarray
    ds = (
        df_avg[["datetime", "latitude", "xco2", "xco2_uncertainty"]]
        .set_index(["datetime", "latitude"])
        .to_xarray()
    )

    ds = ds.rename({"latitude": "lat", "datetime": "time"})

    ds = ds.sortby(["lat", "time"])

    # interpolate latitude
    ds_interp = ds.interp(lat=np.arange(-89.5, 90, 1))

    # this works
    # Create the new longitude dimension
    lon = np.arange(0.5, 360, 1)
    lon_da = xr.DataArray(lon, dims="lon")

    # Broadcast data across the new dimension
    ds_with_lon = ds_interp.broadcast_like(lon_da).assign_coords(
        lon=lon_da
    )  # .assign_coords(lon) #.expand_dims(lon=('lon', lon))

    ds_final = ds_with_lon.transpose("time", "lat", "lon")

    ds_final.to_netcdf(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="path to raw data",
        default="/home/ljg48/palmer_scratch/data/noaa-mbl/raw/co2_GHGreference.1749290223_surface.txt",
    )
    parser.add_argument(
        "--output",
        help="path to save files to",
        default="/home/ljg48/palmer_scratch/data/noaa-mbl/processed/noaa-mbl_197901-202301_1x1.nc",
    )
    args = parser.parse_args()

    input = Path(args.input)
    output = Path(args.output)

    main(input=input, output=output)

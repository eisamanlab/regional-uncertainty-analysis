import argparse
from pathlib import Path
import logging

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


def main(data_dir: str, output_file: str, start_year: int, end_year: int):
    """merge GCB models"""
    path = Path(data_dir)

    # which years to get data for
    list_of_years = range(start_year, end_year + 1)

    list_of_ds = []

    for file in path.glob("GCB-2023_dataprod_*.nc"):
        product_name = file.stem.split("_")[2]

        logging.info(f"Processing: {product_name}")

        _ds = (
            xr.open_dataset(file)
            .pipe(lambda x: x.sel(time=x.time.dt.year.isin(list_of_years)))["sfco2"]
            .to_dataset()
            .assign_coords(product=product_name)
            .expand_dims(dim="product")
        )

        list_of_ds.append(_ds)
        del _ds

    ds = xr.merge(list_of_ds)
    ds.to_netcdf(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        help="data directory",
        default="/home/ljg48/project/data/GCB2023/raw/",
    )
    parser.add_argument(
        "--output",
        help="path to CSV file to import",
        default="/home/ljg48/project/data/GCB2023/processed/GCB23_dataprod_merged.nc",
    )

    parser.add_argument("--start-year", help="starting year", default=1982)

    parser.add_argument("--end-year", help="ending year", default=2022)

    parser.add_argument(
        "--log-file", help="path to log file", default="./process-gcb-2023-dataprods.log"
    )

    args = parser.parse_args()

    DATA_DIR = args.input
    OUTPUT_FILE = args.output
    LOG_FILE = args.log_file
    START_YEAR = int(args.start_year)
    END_YEAR = int(args.end_year)

    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    main(
        data_dir=DATA_DIR,
        output_file=OUTPUT_FILE,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )
    logging.info("Done")

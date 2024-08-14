import argparse
from datetime import datetime
import logging
from pathlib import Path
import time

import dask
from siphon.catalog import TDSCatalog
import xarray as xr
import zarr


def tds_catalog(url):
    """create a catalog client

    Parameters
    ---------
    url: str
        path to THREDDS data catalog

    Returns
    ---------
    catalog: TDSCatalog
        TDS catalog client

    Example
    ---------
    url = "https://thredds.daac.ornl.gov/thredds/catalog/ornldaac/1741/catalog.xml"
    catalog = tds_catalog(url)
    """
    return TDSCatalog(url)


def tds_generator(catalog):
    """generator for files in a THREDDS catalog

    Parameters
    ---------
    catalog: TDSCatalog
        TDS catalog client

    Returns
    ---------
    dataset: str
        name of the dataset on server

    Example
    ---------
    url = "https://thredds.daac.ornl.gov/thredds/catalog/ornldaac/1741/catalog.xml"
    catalog = tds_catalog(url)
    gen = tds_generator(catalog)
    datasets = [fl for fl in gen]
    """
    yield from (dataset for dataset in catalog.datasets)


def get_dataset_url(catalog, dataset):
    """retrieve the URL to the dataset

    Parameters
    ---------
    catalog: TDSCatalog
        TDS catalog client

    dataset: str
        name of the dataset in the catalog you want to open

    Returns
    ---------
    dataset: str
        name of the dataset on server

    Example
    ---------
    url = "https://thredds.daac.ornl.gov/thredds/catalog/ornldaac/1741/catalog.xml"
    catalog = tds_catalog(url)
    gen = tds_generator(catalog)
    datasets = [fl for fl in gen]
    dataset = datasets[0]
    dataset_url = get_dataset_url(catalog, dataset)
    """
    return catalog.datasets[dataset].access_urls["OPENDAP"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_year",
        help="first year to download",
        default=1982,
    )
    parser.add_argument(
        "--end_year",
        help="final year to download",
        default=2023,
    )
    parser.add_argument(
        "--file_path",
        help="output file data is saved to",
        default="./oisst_links.txt",
    )
    args = parser.parse_args()

    BASE_URL = f"https://www.ncei.noaa.gov/thredds/catalog/OisstBase/NetCDF/V2.1/AVHRR"
    CATALOG_URL = f"{BASE_URL}/catalog.xml"

    catalog = TDSCatalog(CATALOG_URL)

    subdirs = catalog.catalog_refs
    subdirs_filtered = [
        subdir
        for subdir in subdirs
        if int(subdir[:4]) >= args.start_year and int(subdir[:4]) <= args.end_year
    ]

    for subdir in subdirs_filtered:
        sub_catalog_url = f"{BASE_URL}/{subdir}/catalog.xml"
        catalog = TDSCatalog(sub_catalog_url)
        datasets = catalog.datasets
        dataset_urls = [
            catalog.datasets[dataset].access_urls["HTTPServer"] for dataset in datasets
        ]

        with open(args.file_path, "a") as file:
            for dataset_url in dataset_urls:
                file.write(dataset_url + "\n")


if __name__ == "__main__":
    main()

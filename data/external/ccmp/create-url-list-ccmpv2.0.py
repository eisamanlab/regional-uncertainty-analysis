#!/usr/bin/env python
""" 
create list of ccmp files to download
python create-url-list.py --start-year <year> --end-year <year> --output-path <path>
"""
import argparse
from pathlib import Path
import requests

from siphon.catalog import TDSCatalog
import xarray as xr

def download_data(year_range, output_path):
    output_list = []

    # THREDDS catalog URL
    base_tds_url = "https://thredds.rda.ucar.edu"
    endpoint = "/thredds/catalog/files/g/ds745.1/ccmp_6hourly/"
    catalog_url = f"{base_tds_url}{endpoint}catalog.xml"

    try:
        # create catalog obj
        catalog = TDSCatalog(catalog_url)
    except requests.exceptions.HTTPError as e:
        print("!! HTTPError:", e)

    # select years and list of sub catalogs
    years_to_include = [str(year) for year in range(year_range[0], year_range[1]+1)]
    sub_cats = [sub_catalog for sub_catalog in catalog.catalog_refs if any(sub_catalog.startswith(year) for year in years_to_include)]

    # reference for subcategory
    for sub_cat in sub_cats:
        sub_cat_url = f"{base_tds_url}{endpoint}{sub_cat}/catalog.xml"

        # sub_catalog obj
        sub_catalog = TDSCatalog(sub_cat_url)

        # get dataset
        for dataset in sub_catalog.datasets:
            dataset_url = sub_catalog.datasets[dataset].access_urls["HTTPServer"]
            output_list.append(dataset_url)

    with open(output_path, 'w') as f:
        for line in output_list:
            f.write(f"{line}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from CCMP THREDDS catalog")
    parser.add_argument("--start-year", type=int, required=True, help="Start year of the range")
    parser.add_argument("--end-year", type=int, required=True, help="End year of the range")
    parser.add_argument("--output-file", type=str, default='./file-list.txt', help="Output path for the downloaded files")
    args = parser.parse_args()

    year_range = (args.start_year, args.end_year)
    download_data(year_range, args.output_file)

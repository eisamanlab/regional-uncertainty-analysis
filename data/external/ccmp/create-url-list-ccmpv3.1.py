#!/usr/bin/env python
""" 
create list of ccmp files to download
python create-url-list.py --start-year <year> --end-year <year> --output-path <path>
"""
import argparse
import datetime
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse


def ccmp_31_nc_daily_filename(*, base: str, date_to_get: datetime.date) -> str:
    """Return the filename of the CCMP 3.1 netCDF file for the given date."""

    date_path = f"Y{date_to_get:%Y}/M{date_to_get:%m}/CCMP_Wind_Analysis_{date_to_get:%Y%m%d}_V03.1_L4.nc"

    if urlparse(base).scheme in ('http', 'https', 'ftp'):
        return urljoin(base + '/', date_path)
    else:
        base_path = Path(base)
        return str(base_path.joinpath(date_path))


def download_data(year_range, output_path):
    """this just gets the list of urls to download the data"""
    
    base = 'https://data.remss.com/ccmp/v03.1/'

    start_year = year_range[0]
    end_year = year_range[1]
    
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(end_year, 12, 31)
    
    date_list = []
    
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += datetime.timedelta(days=1)
            
    url_generator= (ccmp_31_nc_daily_filename(base=base, date_to_get=date) for date in date_list)
    
    with open(output_path, 'w') as file:
        for url in url_generator:
            file.write(url + '\n')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from CCMP THREDDS catalog")
    parser.add_argument("--start-year", type=int, required=True, help="Start year of the range")
    parser.add_argument("--end-year", type=int, required=True, help="End year of the range")
    parser.add_argument("--output-file", type=str, default='./file-list.txt', help="Output path for the downloaded files")
    args = parser.parse_args()

    year_range = (args.start_year, args.end_year)
    download_data(year_range, args.output_file)

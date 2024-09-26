import logging
from pathlib import Path
import warnings

import intake
import xarray as xr

# Suppress FutureWarning from xarray
# To access a mapping from dimension names to lengths, please use `Dataset.sizes`
warnings.filterwarnings("ignore", category=FutureWarning, module="intake_xarray")

def main():
    # output path
    output_path = Path("/home/ljg48/project/data/oae-uncertainty-data/")
    output_file = output_path / "wind-1x1-1993-2022.nc"
    
    # Load the master catalog of raw data
    # data is for each year
    cat = intake.open_catalog("/home/ljg48/project/oae-uncertainty/data/intake/master.yaml")
    
    fl_era5_wind = output_path / "era5-wind-1x1-1993-2022.nc"
    fl_ccmp_wind = output_path / "ccmp-1x1-1993-2022.nc"
    fl_jra3q_wind = output_path / "jra3q-wind-1x1-1993-2022.nc"
    
    list_of_files = [fl_era5_wind, fl_ccmp_wind, fl_jra3q_wind]
    
    list_of_ds = []
    
    for file in list_of_files:
        product_name = file.stem.split("-")[0]
    
        #logging.info(f"Processing: {product_name}")
    
        _ds = (
            xr.open_dataset(file)
            #.pipe(lambda x: x.sel(time=x.time.dt.year.isin(list_of_years)))["sfco2"]
            #.to_dataset()
            .assign_coords(product=product_name)
            .expand_dims(dim="product")
        )
    
        list_of_ds.append(_ds)
        del _ds
    
    ds = xr.merge(list_of_ds)
    ds.to_netcdf(output_file)


if __name__ == "__main__":
    logging.basicConfig(
        filename="./output.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    main()
    logging.info("Done")
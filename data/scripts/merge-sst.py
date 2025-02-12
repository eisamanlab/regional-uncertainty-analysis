import logging
from pathlib import Path
import warnings

import intake
import xarray as xr

# Suppress FutureWarning from xarray
# To access a mapping from dimension names to lengths, please use `Dataset.sizes`
warnings.filterwarnings("ignore", category=FutureWarning, module="intake_xarray")

def main():
    # input path
    input_path = Path("/home/ljg48/project/oae-uncertainty/data/processed")
    
    # output path
    output_path = Path("/home/ljg48/project/oae-uncertainty/data/processed")
    output_file = output_path / "sst-1x1-1993-2022.nc"

    # individual data files paths
    fl_oisst = input_path / "oisst-1x1-1993-2022.nc"
    fl_ostia = input_path / "ostia-1x1-1993-2022.nc"
    fl_cobe = input_path / "cobe2-1x1-1993-2022.nc"
    fl_esa = input_path / "esa-1x1-1993-2022.nc"
    
    list_of_files = [fl_oisst, fl_ostia, fl_cobe, fl_esa]
    
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
    
    ds = xr.merge(list_of_ds)[['sst']]
    ds.to_netcdf(output_file)


if __name__ == "__main__":
    logging.basicConfig(
        filename="./output.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    main()
    logging.info("Done")
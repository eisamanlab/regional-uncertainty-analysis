import logging
from pathlib import Path
import warnings

import numpy as np
import xarray as xr

def main():
    # output path
    output_path = Path("/home/ljg48/project/data/oae-uncertainty-data/")
    output_file = output_path / "pco2-1x1-1993-2022.nc"

    ds_mod = (
        xr.open_dataset("/home/ljg48/project/data/GCB2023/processed/gcb-2023-models_1x1.nc")
        .rename({"model": "product"})
        .set_index({"product": "product"})
        .sel(time=slice('1993', '2022'))
    )
    
    ds_prod = (
        xr.open_dataset("/home/ljg48/project/data/GCB2023/processed/gcb-2023-dataprods_1x1.nc")
        .sel(time=slice('1993', '2022'))
        .assign_coords(time = lambda _ds: _ds.time + np.timedelta64(14, 'D'))
    )
    
    ds = xr.concat([ds_mod, ds_prod], dim='product')

    ds.to_netcdf(output_file)


if __name__ == "__main__":
    logging.basicConfig(
        filename="./output.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    main()
    logging.info("Done")
    
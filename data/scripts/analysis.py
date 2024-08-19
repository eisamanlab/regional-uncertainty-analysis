"""
This script calculates fractional uncertanties.
author: Luke Gloege
"""

import logging
from pathlib import Path
import warnings

import xarray as xr

from fluxerror import solubility as sol
from fluxerror import gas_transfer_velocity as kw
from fluxerror import delta_pco2 as dpco2

def main():
    # data path
    data_path = Path("/home/ljg48/project/data/oae-uncertainty-data/")
    output_file = data_path / "fractional-uncertanties-1x1-1993-2022.nc"
    
    # read data    
    ds_sst = xr.open_dataset(data_path / "oisst-1x1-1993-2022.nc")
    ds_sss = xr.open_dataset(data_path / "en4-1x1-1993-2022.nc")
    ds_wind = xr.open_dataset(data_path / "wind-1x1-1993-2022.nc")
    ds_pco2 = xr.open_dataset(data_path / "pco2-1x1-1993-2022.nc")

    # dictionary of input arguments
    input_kwargs = dict(
        temp_C = ds_sst['sst'],
        S = ds_sss['salinity'],
        delta_T = ds_sst['err'],
        delta_S = ds_sss['salinity_uncertainty'],
        u_mean = ds_wind['ws_mean'].mean("product"),
        u_std = ds_wind['ws_std'].mean("product"),
        delta_umean = ds_wind['ws_mean'].std("product"),
        delta_ustd = ds_wind['ws_std'].std("product"),
        pco2 = ds_pco2['sfco2'].mean('product'),
        delta_pco2 = ds_pco2['sfco2'].std('product'),
    )

    # perform calculations
    frac_sol_sss = sol.weiss1974.frac.ko_wrt_salt(**input_kwargs).to_dataset(name="frac_sol_sss")
    frac_sol_sst = sol.weiss1974.frac.ko_wrt_temp(**input_kwargs).to_dataset(name="frac_sol_sst")
    frac_kw_umean = kw.wanninkhof2014.frac.kw_umean(**input_kwargs).to_dataset(name="frac_kw_umean")
    frac_kw_ustd = kw.wanninkhof2014.frac.kw_ustd(**input_kwargs).to_dataset(name="frac_kw_ustd")
    frac_kw_sc = kw.wanninkhof2014.frac.kw_sc(**input_kwargs).to_dataset(name="frac_kw_sc")
    frac_pco2 = dpco2.frac.pco2ocn(**input_kwargs).to_dataset(name="frac_pco2")

    # merge all the variables
    ds = xr.merge([
        frac_sol_sss,
        frac_sol_sst,
        frac_kw_umean,
        frac_kw_ustd,
        frac_kw_sc,
        frac_pco2,
    ])

    ds.to_netcdf(output_file)


if __name__ == "__main__":
    logging.basicConfig(
        filename="./output.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    main()
    
    logging.info("Done")
    
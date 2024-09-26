import numpy as np
from siphon.catalog import TDSCatalog
import xarray as xr
import xesmf as xe


def download_lsm(access_url: str = "OpenDAP") -> xr.Dataset:
    """download lsm rom UCAR RDA as xarray dataset"""
    base_tds_url = "https://thredds.rda.ucar.edu"
    endpoint = "/thredds/catalog/files/g/ds633.0/e5.oper.invariant/197901/"
    catalog_url = f"{base_tds_url}{endpoint}catalog.xml"

    catalog = TDSCatalog(catalog_url)

    variables = ["e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc"]
    datasets = [
        dset for dset in catalog.datasets if any(var in dset for var in variables)
    ]

    dataset_url = catalog.datasets[datasets[0]].access_urls[access_url]

    ds = (
        xr.open_dataset(dataset_url)
        .squeeze()
        .sortby(["longitude", "latitude"])
        .rename({"latitude": "lat", "longitude": "lon", "LSM": "lsm"})
    )
    return ds


def grid(lon_min=-89.5, lon_max=90, lat_min=0.5, lat_max=359.5, dlon=1.0, dlat=1.0):
    """create grid to regrid to"""
    return xr.Dataset(
        {
            "lat": (["lat"], np.arange(-89.5, 90, 1.0), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(0.5, 359.5, 1.0), {"units": "degrees_east"}),
        }
    )


def limit_lat_range(ds, lat_min=-67, lat_max=67):
    ds_copy = ds.copy()
    lat = ds_copy["lat"]
    lon = ds_copy["lon"]
    return ds_copy.where((lat >= lat_min) & (lat <= lat_max))


def mask_inland_seas(ds):
    ds_copy = ds.copy()
    lat = ds_copy["lat"]
    lon = ds_copy["lon"]
    return ds_copy.where(~((lat > 24) & (lat < 70) & (lon > 14) & (lon < 70)))


def mask_hudson_bay(ds):
    ds_copy = ds.copy()
    lat = ds_copy["lat"]
    lon = ds_copy["lon"]
    return ds_copy.where(
        ~((lat > 50) & (lat < 70) & (lon > -100 + 360) & (lon < -70 + 360))
    ).where(~((lat > 70) & (lat < 80) & (lon > -130 + 360) & (lon < -80 + 360)))


def mask_red_sea(ds):
    ds_copy = ds.copy()
    lat = ds_copy["lat"]
    lon = ds_copy["lon"]
    return ds_copy.where(~((lat > 10) & (lat < 25) & (lon > 10) & (lon < 45))).where(
        ~((lat > 20) & (lat < 50) & (lon > 0) & (lon < 20))
    )


def mask_great_lakes(ds):
    ds_copy = ds.copy()
    lat = ds_copy["lat"]
    lon = ds_copy["lon"]
    return ds_copy.where(~((lat > 44) & (lat < 60) & (lon > 250) & (lon < 278)))


if __name__ == "__main__":
    OUTPUT_DIR = "/home/ljg48/project/data/lsm/processed"

    ds = download_lsm()

    ds_grid = grid()

    regridder = xe.Regridder(ds, ds_grid, method="bilinear")
    ds_regrid = regridder(ds, keep_attrs=True).compute()

    # need to add one since lsm returns 0 for ocean points
    ds_out = ds_regrid.assign(
        mask=ds_regrid.where(ds_regrid["lsm"] == 0, np.nan)["lsm"] + 1
    )

    ds_tmp = limit_lat_range(ds_out)
    ds_tmp = mask_inland_seas(ds_tmp)
    ds_tmp = mask_hudson_bay(ds_tmp)
    ds_tmp = mask_red_sea(ds_tmp)
    ds_tmp = mask_great_lakes(ds_tmp)

    # need to delete this property in order to save to netcdf
    # this is probably the solution if you get attribute error: string match name in use
    _ = ds_tmp.attrs.pop("_NCProperties")
    ds_tmp.to_netcdf(f"{OUTPUT_DIR}/ocean-mask_invariant_1x1.nc")

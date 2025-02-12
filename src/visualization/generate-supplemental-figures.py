import itertools
import os
from pathlib import Path
import warnings


from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
import cmocean as cm
import fluxerror
import intake
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import LineCollection
import numpy as np
import pyseaflux as sf
import xarray as xr

from plotting import (
    create_grid,
    add_coastline,
    add_continents,
    add_colorbar,
    add_colorbar_to_subplot,
    add_title,
    plot_data
)


def xr_add_cyclic_point(data, cyclic_coord="lon"):
    '''
    cyclic_point : a wrapper for catopy's apply_ufunc

    Inputs
    =============
    data         : dataSet you want to add cyclic point to
    cyclic_coord : coordinate to apply cyclic to

    Returns
    =============
    cyclic_data : returns dataset with cyclic point added

    '''
    data = data.assign_coords(lon = (((data.lon + 180) % 360) - 180)).sortby('lon')
    
    data = xr.apply_ufunc(add_cyclic_point, data.load(),
                          input_core_dims=[[cyclic_coord]], 
                          output_core_dims=[['tmp_new']]).rename({'tmp_new': cyclic_coord})
    
    data = data.roll(lon=-180)
    
    return data
    
def add_colorbar_each(grid, ind,  vmin, vmax, cmap = cm.cm.amp, ncolors=101, label='', **kwargs):
    bounds = np.linspace(vmin, vmax, ncolors) 
    params_default = {
        'orientation':'horizontal',
        'cmap': cmap,
        "norm": mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        #"norm": mpl.colors.BoundaryNorm(bounds, ncolors),
        'extend': None
    }
    params = {**params_default, **kwargs}
    col = mpl.colorbar.ColorbarBase(grid.cbar_axes[ind], **params)
    return col
    #col.ax.set_xlabel(label)



def _add_xarray_attrs(func):
    """A helper function to add attributes to xarray."""
    import re

    from functools import wraps

    from xarray import DataArray

    def get_refs(func):
        """gets the reference from the docs where the reference is formatted
        as \nReferences: ..."""
        found = re.findall("References:(.*)", func.__doc__, flags=re.DOTALL)
        if any(found):
            ref = " ".join([s.strip() for s in found[0].split("\n")]).strip()
            return ref
        else:
            return ""

    def get_code(func):
        """get the formulation of kw from the function code. Requires the line
        to start with k = ..."""
        import inspect

        raw = "".join(inspect.getsource(func))
        found = re.findall("(k = .*)", raw)

        if any(found):
            code = found[0]
            return code
        else:
            return ""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """wrapper that adds the xarray metadata if input is xarray"""
        out = func(*args, **kwargs)
        if isinstance(out, DataArray):
            # full reference based on function name. This is manually added
            names = {
                "k_Li86": "Liss and Merlivat (1986)",
                "k_Wa92": "Wanninkhof (1992)",
                "k_Wa99": "Wanninkhof and McGillis(1999)",
                "k_Ni00": "Nightingale et al. (2000)",
                "k_Mc01": "McGillis et al (2001)",
                "k_Ho06": "Ho et al. (2006)",
                "k_Sw07": "Sweeney et al. (2007)",
                "k_Wa09": "Wanninkhof et al. (2009)",
                "k_Wa14": "Wanninkhof et al. (2014)",
            }
            name = names[func.__name__]
            out = out.assign_attrs(
                units="cm/hr",
                description=f"gas transfer velocity of CO2 in seawater using {name}",
                reference=get_refs(func),
                formulation=get_code(func),
            )
        return out

    return wrapper

def schmidt_number(temp_C):
    """
    Calculates the Schmidt number as defined by Jahne et al. (1987) and listed
    in Wanninkhof (2014) Table 1.

    Args:
        temp_C (array): temperature in degrees C

    Returns:
        array: Schmidt number (dimensionless)

    Examples:
        >>> schmidt_number(20)  # from Wanninkhof (2014)
        668.344

    References:
        Jähne, B., Heinz, G., & Dietrich, W. (1987). Measurement of the
        diffusion coefficients of sparingly soluble gases in water. Journal
        of Geophysical Research: Oceans, 92(C10), 10767–10776.
        https://doi.org/10.1029/JC092iC10p10767

    Note: code from seaflux package
    """
    from numpy import nanmedian

    if nanmedian(temp_C) > 270:
        raise ValueError("temperature is not in degC")

    T = temp_C

    a = +2116.8
    b = -136.25
    c = +4.7353
    d = -0.092307
    e = +0.0007555

    Sc = a + b * T + c * T ** 2 + d * T ** 3 + e * T ** 4

    return Sc

@_add_xarray_attrs
def k_Wa14(wind_second_moment, temp_C):
    """
    Calculates the gas transfer coeffcient for CO2 using the formulation
    of Wanninkhof et al. (2014)

    The gas transfer velocity has been scaled for the Cross-Calibrated Multi-
    Platform (CCMP) Winds product. Note that using this function for any other
    wind product is not correct.

    .. math::
        k_{660} = 0.251 \\cdot U^2

    Args:
        wind_second_moment (array): wind speed squared in m2/s2. Note that the
            second moment should be calculated at the native resolution of the
            wind to avoid losses of variability when taking the square product.
        temp_C (array): temperature in degrees C

    Returns:
        kw (array): gas transfer velocity (k660) in cm/hr

    References:
        Wanninkhof, R. H. (2014). Relationship between wind speed and gas
        exchange over the ocean revisited. Limnology and Oceanography:
        Methods, 12(JUN), 351–362. https://doi.org/10.4319/lom.2014.12.351

    Note: code from seaflux package
    """

    U2 = wind_second_moment

    Sc = schmidt_number(temp_C)
    k = 0.251 * U2 * (660 / Sc) ** 0.5

    return k


def solubility_weiss1974(salt, temp_C, press_atm=1, checks=True):
    """Calculates the solubility of CO2 in sea water

    Used in the calculation of air-sea CO2 fluxes. We use the formulation by
    Weiss (1974) summarised in Wanninkhof (2014).

    Args:
        salt (array): salinity in PSU
        temp_K (array): temperature in deg Kelvin
        press_atm (array): pressure in atmospheres. Used in the solubility
            correction for water vapour pressure. If not given, assumed
            that press_atm is 1atm

    Returns:
        array: solubility of CO2 in seawater (:math:`K_0`) in mol/L/atm

    Examples:
        from Weiss (1974) Table 2 but with pH2O correction

        >>> solubility_weiss1974(35, 299.15)
        0.029285284543519093

    Note: code from seaflux package
    """

    from numpy import exp, log, nanmedian
    from xarray import DataArray

    #from . import vapour_pressure as vapress

    #if checks:
    #    if nanmedian(temp_K) < 270:
    #        raise ValueError("Temperature is not in Kelvin")

    temp_K = temp_C + 273.15
    T = temp_K
    S = salt
    P = press_atm

    # from table in Wanninkhof 2014
    a1 = -58.0931
    a2 = +90.5069
    a3 = +22.2940
    b1 = +0.027766
    b2 = -0.025888
    b3 = +0.0050578

    T100 = T / 100
    K0 = exp(
        a1 + a2 * (100 / T) + a3 * log(T100) + S * (b1 + b2 * T100 + b3 * T100 ** 2)
    )

    #pH2O = vapress.weiss1980(S, T)
    #K0 = K0 / (P - pH2O)

    # mol / L / atm --> mol / m3 / uatm
    # mol . L-1 . atm-1 * (L . m-3) * (atm . uatm-1)
    #                       1000    *   1e-6

    if isinstance(K0, DataArray):
        K0 = K0.assign_attrs(
            units="mol/L/atm",
            description=(
                "solubility based on Weiss (1974)"
            ),
        )

    return K0  # units mol/L/atm



def generate_figure_pco2_variables(figdir, filename="uncertainty-variables.png", savefig=True):

    data_dir = "/home/ljg48/project/oae-uncertainty/data/processed"
    
    ds_sst = xr.open_dataset(f"{data_dir}/sst-1x1-1993-2022.nc")
    ds_sss = xr.open_dataset(f"{data_dir}/salinity-1x1-1993-2022.nc")
    ds_wind = xr.open_dataset(f"{data_dir}/wind-1x1-1993-2022.nc")
    ds_pco2 = xr.open_dataset(f"{data_dir}/pco2-1x1-1993-2022.nc")
    ds_mask = xr.open_dataset(f"{data_dir}/ocean-mask_invariant_1x1.nc")
    
    ds_sst = xr_add_cyclic_point(ds_sst)
    ds_sss = xr_add_cyclic_point(ds_sss)
    ds_wind = xr_add_cyclic_point(ds_wind)
    ds_pco2 = xr_add_cyclic_point(ds_pco2)
    ds_mask = xr_add_cyclic_point(ds_mask)

    vmin = 0
    vmax = 1
    cmap = mpl.cm.afmhot_r #cm.cm.haline
    
    continent_color = [0.6,0.6,0.6]
    
    fig = plt.figure(dpi=200)
    grid = create_grid(fig, axes_pad=0.8, nrows_ncols=(2, 2), cbar_size='5%', cbar_mode='each' )
    
    ind = 0
    data = ds_sst['sst'].std("product").mean("time").where(ds_mask['mask'] == 1)
    grid = add_continents(grid, ind, facecolor=continent_color)
    plot_data(grid, ind, data, vmin=vmin, vmax=vmax, cmap=cmap)
    add_title(grid, ind, title=r'Standard deviation across SST products', fontsize=8)
    
    col = add_colorbar_each(grid, ind, vmin=vmin, vmax=vmax, cmap=cmap, extend="max")
    col.ax.set_xlabel(r'SST uncertainty [$^{\circ}C]$', fontsize=8)
    
    grid[ind].text(0.0, 1, 'A', transform=grid[ind].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    
    
    ind = 1
    data = ds_sss['salinity'].std("product").mean("time").where(ds_mask['mask'] == 1)
    grid = add_continents(grid, ind, facecolor=continent_color)
    plot_data(grid, ind, data, vmin=vmin, vmax=vmax, cmap=cmap)
    add_title(grid, ind, title=r'Standard deviation across salinity products', fontsize=8)
    
    
    # Add rivers
    rivers_shp = shapereader.natural_earth(resolution='110m',
                                           category='physical',
                                           name='rivers_lake_centerlines')
    
    # Create a feature for rivers with custom linewidth
    rivers_feature = cfeature.ShapelyFeature(shapereader.Reader(rivers_shp).geometries(),
                                             ccrs.PlateCarree(), facecolor='none', edgecolor='blue')
    
    # Add rivers with a thin linewidth
    grid[ind].add_feature(rivers_feature, linewidth=0.5)  # Adjust linewidth as desired
    
    
    
    col = add_colorbar_each(grid, ind, vmin=vmin, vmax=vmax, cmap=cmap, extend="max")
    col.ax.set_xlabel('SSS uncertainty [PSU]', fontsize=8)
    
    grid[ind].text(0.0, 1, 'B', transform=grid[ind].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    
    ind = 2
    data = ds_wind['ws_mean'].std("product").mean("time").where(ds_mask['mask'] == 1)
    grid = add_continents(grid, ind, facecolor=continent_color)
    plot_data(grid, ind, data, vmin=vmin, vmax=vmax, cmap=cmap)
    add_title(grid, ind, title=r'Standard deviation across wind products', fontsize=8)
    
    col = add_colorbar_each(grid, ind, vmin=vmin, vmax=vmax, cmap=cmap, extend="max")
    col.ax.set_xlabel(r'wind speed uncertainty [m/s]', fontsize=8)
    
    grid[ind].text(0.0, 1, 'C', transform=grid[ind].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    
    
    ind = 3
    data = ds_pco2['sfco2'].std("product").mean("time").where(ds_mask['mask'] == 1)
    grid = add_continents(grid, ind, facecolor=continent_color)
    plot_data(grid, ind, data, vmin=vmin, vmax=50, cmap=cmap)
    add_title(grid, ind, title=r'Standard deviation arcross models/products', fontsize=8)
    
    col = add_colorbar_each(grid, ind, vmin=vmin, vmax=50, cmap=cmap, extend="max")
    col.ax.set_xlabel(r'pCO$_2$ uncertainty [$\mu$atm]', fontsize=8)
    
    grid[ind].text(0.0, 1, 'D', transform=grid[ind].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    
    
    if savefig:
        plt.savefig(figdir / filename, bbox_inches='tight')
        

def generate_figure_pco2_uncertainty(figdir, filename="uncertainty-pco2.png", savefig=True):

    data_dir = "/home/ljg48/project/oae-uncertainty/data/processed"
    
    ds_sst = xr.open_dataset(f"{data_dir}/sst-1x1-1993-2022.nc")
    ds_sss = xr.open_dataset(f"{data_dir}/salinity-1x1-1993-2022.nc")
    ds_wind = xr.open_dataset(f"{data_dir}/wind-1x1-1993-2022.nc")
    ds_pco2 = xr.open_dataset(f"{data_dir}/pco2-1x1-1993-2022.nc")
    ds_mask = xr.open_dataset(f"{data_dir}/ocean-mask_invariant_1x1.nc")
    
    ds_sst = xr_add_cyclic_point(ds_sst)
    ds_sss = xr_add_cyclic_point(ds_sss)
    ds_wind = xr_add_cyclic_point(ds_wind)
    ds_pco2 = xr_add_cyclic_point(ds_pco2)
    ds_mask = xr_add_cyclic_point(ds_mask)

    models = ['ACCESS', 'CESM', 'CNRM', 'FESOM2', 'IPSL', 'MOM6', 'MPIOM', 'MRI', 'NEMO', 'NorESM']
    prods = ['CMEMS-', 'JENA-M', 'JMA-ML', 'LDEO-H', 'NIES-M', 'OceanS', 'SOM-FF', 'UoEX']
    
    continent_color = [0.6,0.6,0.6]
    vmin = 0
    vmax = 50
    cmap = mpl.cm.afmhot_r #cm.cm.haline
    
    fig = plt.figure(dpi=200)
    grid = create_grid(fig, axes_pad=0.2, nrows_ncols=(1, 2), cbar_size='5%')
    
    ind = 0
    data = ds_pco2['sfco2'].where(ds_pco2['product'].isin(models)).std("product").mean("time").where(ds_mask['mask'] == 1)
    grid = add_continents(grid, ind, facecolor=continent_color)
    plot_data(grid, ind, data, vmin=vmin, vmax=vmax, cmap=cmap)
    add_title(grid, ind, title=r'Standard deviation arcross models', fontsize=8)
    
    #col = add_colorbar(grid, vmin=vmin, vmax=vmax, cmap=cmap)
    #col.ax.set_xlabel('Uncertainty in SSS')
    
    
    # Label 'A'
    grid[ind].text(0.0, 1, 'A', transform=grid[ind].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    
    ind = 1
    data = ds_pco2['sfco2'].where(ds_pco2['product'].isin(prods)).std("product").mean("time").where(ds_mask['mask'] == 1)
    grid = add_continents(grid, ind, facecolor=continent_color)
    plot_data(grid, ind, data, vmin=vmin, vmax=vmax, cmap=cmap)
    add_title(grid, ind, title=r'Standard deviation across products', fontsize=8)
    
    col = add_colorbar(grid, vmin=vmin, vmax=vmax, cmap=cmap, extend="max")
    col.ax.set_xlabel(r'fCO$_2$ uncertainty [$\mu$atm]')
    
    # Label 'B'
    grid[ind].text(0.0, 1, 'B', transform=grid[ind].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    
    
    
    import matplotlib.patches as patches
    
    #grid[ind].add_patch(patches.Rectangle(xy=[70, -45], width=90, height=90,
    #                                facecolor='red',
    #                                alpha=0.2,
    #                                transform=ccrs.Geodetic())
    #                   )
                        
    # Define upwelling regions (approximate latitude/longitude ranges)
    upwelling_regions = [
        #{"name": "California", "lon": (230, 250), "lat": (34, 44)},  # 230-245°E (110-125°W)
        {"name": "Peru", "lon": (270, 290), "lat": (-16, 6)},  # 270-285°E (75-90°W)
        {"name": "Benguela", "lon": (5, 20), "lat": (-35, 0)},  # 10-20°E
        #{"name": "NW Africa", "lon": (335, 355), "lat": (12, 22)},  # 340-355°E (5-20°W)
    ]
    
    # Add white boxes after plotting data
    for ind in range(2):  # Assuming two subplots
        ax = grid[ind]  # Get the correct subplot from the grid
        
        for region in upwelling_regions:
            lon_min, lon_max = region["lon"]
            lat_min, lat_max = region["lat"]
            
            # Create a white box with zorder for visibility
            rect = patches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                     linewidth=0.5, edgecolor='black', facecolor='none',
                                     transform=ccrs.PlateCarree(central_longitude=0),
                                     linestyle='solid', zorder=10)  # Ensure it's on top
            
            ax.add_patch(rect)  # Add to the current subplot
    
    
    if savefig:
        plt.savefig(figdir / filename, bbox_inches='tight')
        


def generate_figure_correlation_variables(figdir, filename="correlation-variables.png", savefig=True):

    continent_color = [0.6,0.6,0.6]
    
    data_dir = "/home/ljg48/project/oae-uncertainty/data/processed"
    
    ds_sst = xr.open_dataset(f"{data_dir}/sst-1x1-1993-2022.nc")
    ds_sss = xr.open_dataset(f"{data_dir}/salinity-1x1-1993-2022.nc")
    ds_wind = xr.open_dataset(f"{data_dir}/wind-1x1-1993-2022.nc")
    ds_pco2 = xr.open_dataset(f"{data_dir}/pco2-1x1-1993-2022.nc")
    ds_mask = xr.open_dataset(f"{data_dir}/ocean-mask_invariant_1x1.nc")
    
    ds_sst = xr_add_cyclic_point(ds_sst)
    ds_sss = xr_add_cyclic_point(ds_sss)
    ds_wind = xr_add_cyclic_point(ds_wind)
    ds_pco2 = xr_add_cyclic_point(ds_pco2)
    ds_mask = xr_add_cyclic_point(ds_mask)

    wind_unc = ds_wind['ws_mean'].std("product").where(ds_mask['mask'] == 1)
    sst_unc = ds_sst['sst'].std("product").where(ds_mask['mask'] == 1)
    sss_unc = ds_sss['salinity'].std("product").where(ds_mask['mask'] == 1)
    pco2_unc = ds_pco2['sfco2'].std("product").where(ds_mask['mask'] == 1)
    
    vmin = 0
    vmax = 1
    cmap = mpl.cm.afmhot_r #cm.cm.haline
    cmap = cm.cm.oxy_r
    
    fig = plt.figure(dpi=300)
    grid = create_grid(fig, axes_pad=0.2, nrows_ncols=(3, 2), cbar_size='5%')
    
    ind = 0
    data = xr.corr(wind_unc, sss_unc, dim='time')
    grid = add_continents(grid, ind, facecolor=continent_color)
    plot_data(grid, ind, data, vmin=vmin, vmax=vmax, cmap=cmap)
    add_title(grid, ind, title=r'corr(wind, salinity)', fontsize=8)
    
    grid[ind].text(0.0, 1.1, 'A', transform=grid[ind].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    
    ind = 1
    data = xr.corr(wind_unc, sst_unc, dim='time')
    grid = add_continents(grid, ind, facecolor=continent_color)
    plot_data(grid, ind, data, vmin=vmin, vmax=vmax, cmap=cmap)
    add_title(grid, ind, title=r'corr(wind, SST)', fontsize=8)
    
    grid[ind].text(0.0, 1.1, 'B', transform=grid[ind].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    
    ind = 2
    data = xr.corr(wind_unc, pco2_unc, dim='time')
    grid = add_continents(grid, ind, facecolor=continent_color)
    plot_data(grid, ind, data, vmin=vmin, vmax=vmax, cmap=cmap)
    add_title(grid, ind, title=r'corr(wind, pCO$_2$)', fontsize=8)
    
    grid[ind].text(0.0, 1.1, 'C', transform=grid[ind].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    
    ind = 3
    data = xr.corr(sst_unc, sss_unc, dim='time')
    grid = add_continents(grid, ind, facecolor=continent_color)
    plot_data(grid, ind, data, vmin=vmin, vmax=vmax, cmap=cmap)
    add_title(grid, ind, title=r'corr(SST, salinity)', fontsize=8)
    
    grid[ind].text(0.0, 1.1, 'D', transform=grid[ind].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    
    ind = 4
    data = xr.corr(sst_unc, pco2_unc, dim='time')
    grid = add_continents(grid, ind, facecolor=continent_color)
    plot_data(grid, ind, data, vmin=vmin, vmax=vmax, cmap=cmap)
    add_title(grid, ind, title=r'corr(SST, pCO$_2$)', fontsize=8)
    
    grid[ind].text(0.0, 1.1, 'E', transform=grid[ind].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    
    ind = 5
    data = xr.corr(sss_unc, pco2_unc, dim='time')
    grid = add_continents(grid, ind, facecolor=continent_color)
    plot_data(grid, ind, data, vmin=vmin, vmax=vmax, cmap=cmap)
    add_title(grid, ind, title=r'corr(salinity, pCO$_2$)', fontsize=8)
    
    grid[ind].text(0.0, 1.1, 'F', transform=grid[ind].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    
    col = add_colorbar(grid, vmin=vmin, vmax=vmax, cmap=cmap)
    col.ax.set_xlabel('correlation between errors')
    
    if savefig:
        plt.savefig(figdir / filename, bbox_inches='tight')
        



def generate_figure_correlation(figdir, filename="correlation-bulk-formula.png", savefig=True):
    continent_color = [0.6,0.6,0.6]
    
    #
    # load data
    #
    ds_sst = xr.open_dataset("/home/ljg48/project/oae-uncertainty/data/processed/sst-1x1-1993-2022.nc")
    ds_sss = xr.open_dataset("/home/ljg48/project/oae-uncertainty/data/processed/salinity-1x1-1993-2022.nc")
    ds_wind = xr.open_dataset("/home/ljg48/project/oae-uncertainty/data/processed/wind-1x1-1993-2022.nc")
    ds_pco2 = xr.open_dataset("/home/ljg48/project/oae-uncertainty/data/processed/pco2-1x1-1993-2022.nc")
    ds_mask_orig = xr.open_dataset(f"/home/ljg48/project/oae-uncertainty/data/processed/ocean-mask_invariant_1x1.nc")
    
    
    #
    # solubility for all combinations
    #
    # Get product names
    sst_products = ds_sst["product"].values
    sss_products = ds_sss["product"].values
    
    # Create an empty list to store results
    solubility_data = []
    
    # Compute solubility for every combination of products
    for sst_prod, sss_prod in itertools.product(sst_products, sss_products):
        sst = ds_sst.sel(product=sst_prod)
        sss = ds_sss.sel(product=sss_prod)
    
        sol = solubility_weiss1974(sss['salinity'], sst['sst'])
        sol = sol.assign_coords(sst_product=sst_prod, sss_product=sss_prod)
        
        solubility_data.append(sol)
    
    ds_solubility = xr.concat(solubility_data, dim="combination").to_dataset(name='sol')
    
    # Add metadata for clarity
    ds_solubility["sol"].attrs = {
        "units": "mol/L/atm",
        "description": "CO2 solubility computed from multiple SST and SSS products"
    }
    
    
    #
    # gas transfer for all combinations
    #
    # Get product names
    sst_products = ds_sst["product"].values
    wind_products = ds_wind["product"].values
    
    # Create an empty list to store results
    kw_data = []
    
    # Compute solubility for every combination of products
    for sst_prod, wind_prod in itertools.product(sst_products, wind_products):
        sst = ds_sst.sel(product=sst_prod)
        wind = ds_wind.sel(product=wind_prod)
    
        wind_second_moment = wind["ws_mean"]**2 + wind["ws_std"]**2
        
        kw = k_Wa14(wind_second_moment, sst["sst"])
        kw = kw.assign_coords(sst_product=sst_prod, wind_product=wind_prod)
        
        kw_data.append(kw)
        
    ds_kw = xr.concat(kw_data, dim="combination").to_dataset(name='kw')
    
    #
    # generate figures
    #
    vmin = 0
    vmax = 1
    cmap = mpl.cm.afmhot_r #cm.cm.haline
    cmap = cm.cm.oxy_r
    
    fig = plt.figure(dpi=300)
    grid = create_grid(fig, axes_pad=0.2, nrows_ncols=(1, 3), cbar_size='5%')
    
    #.where(ds_mask_orig['mask'] == 1)
    kw_unc = ds_kw["kw"].std("combination").where(ds_mask_orig['mask'] == 1)
    sol_unc = ds_solubility['sol'].std("combination").where(ds_mask_orig['mask'] == 1)
    pco2_unc = ds_pco2['sfco2'].std("product").where(ds_mask_orig['mask'] == 1)
    
    ind = 0
    data = xr.corr(kw_unc, sol_unc, dim='time')
    grid = add_continents(grid, ind, facecolor=continent_color)
    plot_data(grid, ind, data, vmin=vmin, vmax=vmax, cmap=cmap)
    add_title(grid, ind, title=r'corr($\delta$k$_w$, $\delta$K$_0$)', fontsize=8)
    
    grid[ind].text(0.0, 1.1, 'A', transform=grid[ind].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    
    ind = 1
    data = xr.corr(kw_unc, pco2_unc, dim='time')
    grid = add_continents(grid, ind, facecolor=continent_color)
    plot_data(grid, ind, data, vmin=vmin, vmax=vmax, cmap=cmap)
    add_title(grid, ind, title=r'corr($\delta$k$_w$, $\delta$fCO$_2$)', fontsize=8)
    
    grid[ind].text(0.0, 1.1, 'B', transform=grid[ind].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    
    ind = 2
    data = xr.corr(sol_unc, pco2_unc, dim='time')
    grid = add_continents(grid, ind, facecolor=continent_color)
    plot_data(grid, ind, data, vmin=vmin, vmax=vmax, cmap=cmap)
    add_title(grid, ind, title=r'corr($\delta$K$_0$, $\delta$pCO$_2$)', fontsize=8)
    
    grid[ind].text(0.0, 1.1, 'C', transform=grid[ind].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    
    col = add_colorbar(grid, vmin=vmin, vmax=vmax, cmap=cmap)
    col.ax.set_xlabel('correlation between errors')
    
    if savefig:
        plt.savefig(figdir / filename, bbox_inches='tight')
    



if __name__ == "__main__":
    figure_dir = Path(os.path.abspath("../../figures"))
    data_dir = "../../data"

    # generate figures
    generate_figure_correlation_variables(figdir = figure_dir)
    generate_figure_pco2_uncertainty(figdir = figure_dir)
    generate_figure_pco2_variables(figdir = figure_dir)
    generate_figure_correlation(figdir = figure_dir)




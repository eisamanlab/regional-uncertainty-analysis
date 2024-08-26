from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
import cmocean as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pyseaflux as sf
import xarray as xr

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
import cmocean as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pyseaflux as sf
import xarray as xr

def create_grid(fig, central_longitude = -149.5, **kwargs):
    """creates grid"""
    projection=ccrs.Robinson(central_longitude=-149.5)
    
    axes_class = (GeoAxes, dict(projection=projection))
    
    default_params = {
            'rect': [1,1,1],           # specifies the location of the grid
            'axes_class': axes_class,  
            'share_all': False,        # xaxis & yaxis of all axes are shared if True
            'nrows_ncols': (2, 2),     # number of rows and cols. e.g., (2,2)
            "ngrids": None,            # number of grids. nrows x ncols if None
            "direction": "row",        # increasing direction of axes number. [row|column]
            'axes_pad': 0.1,           # pad between axes in inches
            'cbar_location': 'bottom', # [left|right|top|bottom]
            'cbar_mode': 'single',     # [None|single|each]
            'cbar_pad': 0.1,           # pad between image axes and colorbar axes
            'cbar_size': '7%',         # size of the colorbar
        }
    
    params = {**default_params, **kwargs}
    
    grid = AxesGrid(fig, **params) 
    return grid


def add_continents(grid, ind, **kwargs):
    params_default = dict(edgecolor='None', facecolor=[0.3,0.3,0.3])
    params = {**params_default, **kwargs}
    
    grid[ind].set_global()
    
    grid[ind].add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', 
                                                     edgecolor=params['edgecolor'], 
                                                     facecolor=params['facecolor']))
    return grid

def add_coastline(grid, ind, **kwargs):
    params_default = {
        'facecolor': [0.25, 0.25, 0.25], 
        'linewidth': 0.1
    }
    params = {**params_default, **kwargs}
    grid[ind].coastlines(**params)
    return grid


def plot_data(grid, ind, data, vmin, vmax, cmap = cm.cm.amp, ncolors=101, **kwargs):
    bounds = np.linspace(vmin, vmax, ncolors) 
    params_default = {
        "cmap": cmap,
        'transform':ccrs.PlateCarree(central_longitude=0),
        "vmin": vmin,
        "vmax": vmax,
        #"norm": mpl.colors.BoundaryNorm(bounds, ncolors)
    }
    params = {**params_default, **kwargs}
    
    transform = ccrs.PlateCarree(central_longitude=0) 
    sub = grid[ind].pcolormesh(data.lon, data.lat, data, **params)

def add_title(grid, ind, title, **kwargs):
    grid[ind].set_title(title, **kwargs)
    
def add_colorbar(grid, vmin, vmax, cmap = cm.cm.amp, ncolors=101, label='', **kwargs):
    bounds = np.linspace(vmin, vmax, ncolors) 
    params_default = {
        'orientation':'horizontal',
        'cmap': cmap,
        "norm": mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        #"norm": mpl.colors.BoundaryNorm(bounds, ncolors),
        'extend': None
    }
    params = {**params_default, **kwargs}
    col = mpl.colorbar.ColorbarBase(grid.cbar_axes[0], **params)
    return col
    #col.ax.set_xlabel(label)


def add_colorbar_to_subplot(grid, ind, vmin, vmax, cmap = cm.cm.amp, ncolors=101, label='', **kwargs):
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



def plot_map(data, plot_label, cmap_label, vrange=[0, 1], cmap=cm.cm.amp, ncolors=101):
    # Define figure dimensions
    fig = plt.figure(dpi=100)
    
    #===========================================
    # Here we are defining our map
    #===========================================
    
    # define projection, I like to center on on -149.5
    # this puts land masses on the edge and does not 
    # break the pacific ocean in a weird way
    projection=ccrs.Robinson(central_longitude=-149.5)
    
    # these are the parmaters for axesgrid
    params_axesgrid = {'rect': [1,1,1], 
                        'axes_class': (GeoAxes, dict(projection=projection)),
                        'share_all': False,
                        'nrows_ncols': (1, 1),
                        'axes_pad': 0.1,
                        'cbar_location': 'bottom',
                        'cbar_mode': 'single', 
                        'cbar_pad': 0.1,
                        'cbar_size': '7%',
                        #'label_mode': '' # passing undefined lable_mode deprecated soon
                      }
    
    
    # Setup axesgrid
    grid = AxesGrid(fig, **params_axesgrid) 
    
    # Force it so it always plots global grid
    grid[0].set_global()
    
    # Add Contintents
    grid[0].add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', 
                                                     edgecolor='None', 
                                                     facecolor=[0.3,0.3,0.3]))
    
    # params for coastline
    params_coastline = {'facecolor': [0.25, 0.25, 0.25], 
                        'linewidth': 0.1}
    
    # add Coastline - I dont do this, but here is how
    #grid[0].coastlines(facecolor=[0.25, 0.25, 0.25], linewidth=0.1)
    
    #===========================================
    # Here we are plotting data on our map
    #===========================================
    vrange = vrange
    ncolors = ncolors
    cmap = cmap
    
    # Set Longitude if none is given
    lon = np.arange(0.5,360,1)
    lat = np.arange(-89.5,90,1)
            
    transform = ccrs.PlateCarree(central_longitude=0)
    bounds = np.linspace(vrange[0], vrange[1], ncolors)
    
    params_pcolormesh = {#'norm': mpl.colors.BoundaryNorm(bounds, ncolors),
                        'transform':ccrs.PlateCarree(central_longitude=0),
                        'cmap': cmap,
                        'vmin': vrange[0],
                        'vmax': vrange[1]}
    
    sub = grid[0].pcolormesh(data.lon, data.lat, data,
                             **params_pcolormesh)
    
    #===========================================
    # Lets add a colorbar
    #===========================================
    params_colbar = {'orientation':'horizontal',
                    'cmap': cmap,
                    'norm':mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1]),
                    'extend': 'max'}
    
    col = mpl.colorbar.ColorbarBase(grid.cbar_axes[0], **params_colbar)
    col.ax.set_xlabel(cmap_label)
    
    #===========================================
    # add a title
    #===========================================
    grid[0].set_title(plot_label)

    # This adjusts the layout to prevent clipping of labels and titles
    plt.tight_layout() 
    
    return fig
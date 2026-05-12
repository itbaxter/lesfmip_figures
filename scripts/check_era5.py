# %%
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from glob import glob
from scipy import stats 


# %%
from utils.regridding import Regridder, create_target_grid
from utils.stats import compute_trend_ols

# %%
# ======================================================================
# Change these
# ======================================================================
# The directory where the main git repo is located (the one that contains the "data" and "scripts" folders)
WDIR = "/project/tas1/itbaxter/for-tiffany/IPCC_figures/lesfmip_figures"

# The grid to regrid to
target_lat = np.arange(-90, 90.1, 1)
target_lon = np.arange(0, 360, 1)

# %%
# Change if you want to use a different file than the one downloaded with get_era5_plevs.py
era5_file = f"{WDIR}/raw_data/Reanalysis/era5_monthly_u_component_of_wind_1979-2023.nc"

era5_uwind = xr.open_dataset(era5_file).rename({"valid_time": "time",
                                                "longitude": "lon",
                                                "latitude": "lat",
                                                "pressure_level": "plev",
                                                }).drop("number").drop("expver").sortby("lat").sel(plev=250, method='nearest')
print(era5_uwind)

# %%
target_grid = create_target_grid(None, 
                                 target_lat=target_lat, 
                                 target_lon=target_lon)

era5_uwind_1deg = Regridder(era5_uwind, target_grid).regrid()
print(era5_uwind_1deg)


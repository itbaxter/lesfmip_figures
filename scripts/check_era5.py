# %%
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import glob as glob
from scipy import stats 
import matplotlib.gridspec as gridspec

# %%
from utils.regridding import Regridder, create_target_grid
from utils.stats import jja, area_weighted_ave, compute_trend_ols 

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
                                                }).drop_vars(["number","expver"]).sortby("lat").sel(plev=200, method='nearest')
print(era5_uwind)

# %%
target_grid = create_target_grid(None, 
                                 target_lat=target_lat, 
                                 target_lon=target_lon)

era5_uwind_1deg = Regridder(era5_uwind, target_grid, method='conservative').regrid()
print(era5_uwind_1deg)

# %%
# N96 grid
target_lat = np.arange(-90, 90.1, 1.25)
target_lon = np.arange(0, 360, 1.875)

target_grid = create_target_grid(None, 
                                 target_lat=target_lat, 
                                 target_lon=target_lon)

era5_uwind_n96 = Regridder(era5_uwind, target_grid, method='conservative').regrid()
print(era5_uwind_n96)

# %%
# 2.5 degree grid 
target_lat = np.arange(-90, 90.1, 2.5)
target_lon = np.arange(0, 360, 2.5)

target_grid = create_target_grid(None, 
                                 target_lat=target_lat, 
                                 target_lon=target_lon)

era5_uwind_2_5deg = Regridder(era5_uwind, target_grid, method='conservative').regrid()
print(era5_uwind_2_5deg)

# %%
damip_aer_uwind_2_5deg = xr.open_mfdataset(glob.glob('/project/tas1/itbaxter/for-tiffany/IPCC_figures/lesfmip_figures/raw_data/hist-aer/71x144/ua/ua_200mb*nc'),
    combine='nested',concat_dim='member_id')
damip_ghg_uwind_2_5deg = xr.open_mfdataset(glob.glob('/project/tas1/itbaxter/for-tiffany/IPCC_figures/lesfmip_figures/raw_data/hist-GHG/71x144/ua/ua_200mb*nc'),
    combine='nested',concat_dim='member_id')
damip_nat_uwind_2_5deg = xr.open_mfdataset(glob.glob('/project/tas1/itbaxter/for-tiffany/IPCC_figures/lesfmip_figures/raw_data/hist-nat/71x144/ua/ua_200mb*nc'),
    combine='nested',concat_dim='member_id')

# %%
era5_eswj_1deg = area_weighted_ave(jja(era5_uwind_1deg.sel(lat=slice(35,45), lon=slice(30, 120))))
era5_eswj_n96 = area_weighted_ave(jja(era5_uwind_n96.sel(lat=slice(35,45), lon=slice(30, 120))))
era5_eswj_2_5deg = area_weighted_ave(jja(era5_uwind_2_5deg.sel(lat=slice(35,45), lon=slice(30, 120))))
damip_aer_eswj_2_5deg = area_weighted_ave(jja(damip_aer_uwind_2_5deg.sel(lat=slice(35,45), lon=slice(30, 120))))
damip_ghg_eswj_2_5deg = area_weighted_ave(jja(damip_ghg_uwind_2_5deg.sel(lat=slice(35,45), lon=slice(30, 120))))
damip_nat_eswj_2_5deg = area_weighted_ave(jja(damip_nat_uwind_2_5deg.sel(lat=slice(35,45), lon=slice(30, 120))))

# %%
era5_eswj_1deg_trend = compute_trend_ols(era5_eswj_1deg['u'].sel(year=slice(1979, 2019)), dim='year')
era5_eswj_n96_trend = compute_trend_ols(era5_eswj_n96['u'].sel(year=slice(1979, 2019)), dim='year')
era5_eswj_2_5deg_trend = compute_trend_ols(era5_eswj_2_5deg['u'].sel(year=slice(1979, 2019)), dim='year')
damip_aer_eswj_2_5deg_trend = compute_trend_ols(damip_aer_eswj_2_5deg['ua'].sel(year=slice(1979, 2019)).compute(), dim='year')
damip_ghg_eswj_2_5deg_trend = compute_trend_ols(damip_ghg_eswj_2_5deg['ua'].sel(year=slice(1979, 2019)).compute(), dim='year')
damip_nat_eswj_2_5deg_trend = compute_trend_ols(damip_nat_eswj_2_5deg['ua'].sel(year=slice(1979, 2019)).compute(), dim='year')

# %%
def rmac(da, year0=1979, year1=2019):
    return da - da.sel(year=slice(year0, year1)).mean('year')

# %%
fig = plt.figure(figsize=(10, 5.5))

gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[3, 1], hspace=0.3, wspace=0.2, left=0.08, right=0.90, top=0.92, bottom=0.10)

ax = fig.add_subplot(gs[0, 0])

ax.fill_between(damip_aer_eswj_2_5deg['year'], rmac(damip_aer_eswj_2_5deg['ua']).min('member_id'), rmac(damip_aer_eswj_2_5deg['ua']).max('member_id'),
                color='silver', alpha=0.8, label='AER spread')

ax.plot(damip_aer_eswj_2_5deg['year'], rmac(damip_aer_eswj_2_5deg['ua'].mean('member_id')), c='gray', label='AER mean')

ax.plot(era5_eswj_1deg['year'], rmac(era5_eswj_1deg['u']), c='k', label=' (1deg)')
ax.plot(era5_eswj_n96['year'], rmac(era5_eswj_n96['u']), c='r', marker='o', ls='--', lw=0.8, label='ERA5 (N96)')
ax.plot(era5_eswj_2_5deg['year'], rmac(era5_eswj_2_5deg['u']), c='b', marker='s', ls=' ', label='ERA5 (2.5deg)')

years = np.arange(1979, 2020)
ax.plot(years, rmac(era5_eswj_1deg_trend['predicted']), c='k', ls='--', lw=0.8, label=f"Trend (1deg)")
ax.plot(years, rmac(era5_eswj_n96_trend['predicted']), c='r', ls='--', lw=0.8, label=f"Trend (N96)")
ax.plot(years, rmac(era5_eswj_2_5deg_trend['predicted']), c='b', ls='--', lw=0.8, label=f"Trend (2.5deg)")
ax.plot(years, rmac(damip_aer_eswj_2_5deg_trend['predicted']).mean('member_id'), c='gray', ls='--', lw=0.8, label=f"AER Trend (2.5deg)")

ax.set_xlabel('Year')
ax.set_xlim([1979, 2023])
ax.set_ylim([-6,6])
ax.minorticks_on()
ax.legend(frameon=False, ncols=3, fontsize=8)
ax.axhline(0, c='silver', ls='--', lw=0.6)
ax.set_title('Zonal wind over(35N-45N, 30E-120E) at 200 hPa')
ax.text(-0.07, 1.0, "a", fontsize=16, transform=ax.transAxes)

ax = fig.add_subplot(gs[0, 1])
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels(['Obs', 'ALL', 'GHG', 'AER', 'NAT'])
ax.set_ylim([-1, 1])
ax.set_xlim([-0.5, 4.5])
ax.set_ylabel('Linear trend')
ax.axhline(0, c='silver', ls='--', lw=0.4)
ax.text(-0.28, 1.0, "b", fontsize=16, transform=ax.transAxes)
ax.minorticks_on()
ax.xaxis.set_tick_params(which='minor', bottom=False)
ax.set_title('Trend of ESWJ at 200 hPa')

ax.scatter([0.1], [10*era5_eswj_1deg_trend['slope'].values], marker='D', facecolor='none', edgecolor='gray')
ax.scatter([0], [10*era5_eswj_n96_trend['slope'].values], marker='D', facecolor='none', edgecolor='gray',)
ax.scatter([-0.1], [10*era5_eswj_2_5deg_trend['slope'].values], marker='D', facecolor='none', edgecolor='gray')

ax.scatter(np.ones(len(damip_nat_eswj_2_5deg_trend['member_id']))*2, [10*damip_ghg_eswj_2_5deg_trend['slope'].values], marker='D', facecolor='none', edgecolor='gray')
ax.scatter(np.ones(len(damip_nat_eswj_2_5deg_trend['member_id']))*3, [10*damip_aer_eswj_2_5deg_trend['slope'].values], marker='D', facecolor='none', edgecolor='gray')
ax.scatter(np.ones(len(damip_nat_eswj_2_5deg_trend['member_id']))*4, [10*damip_nat_eswj_2_5deg_trend['slope'].values], marker='D', facecolor='none', edgecolor='gray')

ax.errorbar([2], [10*damip_ghg_eswj_2_5deg_trend['slope'].mean('member_id').values], 
    yerr=[10*damip_ghg_eswj_2_5deg_trend['slope'].std('member_id').values], 
    ecolor='red',fmt='D',
    )
ax.errorbar([3], [10*damip_aer_eswj_2_5deg_trend['slope'].mean('member_id').values], 
    yerr=[10*damip_aer_eswj_2_5deg_trend['slope'].std('member_id').values], 
    ecolor='red',fmt='D',
    )
ax.errorbar([4], [10*damip_nat_eswj_2_5deg_trend['slope'].mean('member_id').values], 
    [10*damip_nat_eswj_2_5deg_trend['slope'].std('member_id').values], 
    ecolor='red',fmt='D',
    )

plt.savefig(f"{WDIR}/plots/Dong_et_al_2024-ESWJ_timseries_and_trends.png", dpi=300)
plt.savefig(f"{WDIR}/plots/Dong_et_al_2024-ESWJ_timseries_and_trends.pdf", dpi=300)
# %%

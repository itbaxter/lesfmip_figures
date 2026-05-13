"""
Functions to compute statistics, such as linear trends and p-values, for the data used in the LESFMIP figures.

Author: 

Date:   May 12 2026
"""

# Packages
import xarray as xr
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import linregress as _linregress
    
def compute_trend_ols(da_y, da_x=None, dim='time', verbose=False):
    if da_x is None:
        da_x = da_y[dim]

    def _ols_wrapper(y, x, print_summary=False):
        model = sm.OLS(y, sm.add_constant(x))
        results = model.fit()
        if print_summary:
            print(results.summary())
            print(f"OLS regression results: slope={results.params[1]}, p-value={results.pvalues[1]}, intercept={results.params[0]}")
        return results.params[1], results.pvalues[1], results.params[0]

    slope, p_value, intercept = xr.apply_ufunc(
        _ols_wrapper, da_y, da_x,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], [], []],
        vectorize=True,
        # dask='parallelized',
        output_dtypes=[float, float, float],
        kwargs={'print_summary': verbose},
    )
    predicted = da_x * slope + intercept 

    return xr.Dataset({'slope': slope, 'p_value': p_value, 'intercept': intercept, 'predicted': predicted})

def jja(da):
    return da.where(da['time.month'].isin([6, 7, 8]), drop=True).groupby('time.year').mean('time')

def area_weighted_ave(ds):
    if 'lat' not in ds.dims:
        ds = ds.rename({'latitude':'lat','longitude':'lon'})
    coslat = np.cos(np.deg2rad(ds.lat))
    ds,coslat = xr.broadcast(ds,coslat)
    # ds = ds * coslat
    # #return ds.mean(('lat','lon'),skipna=True)
    # return ds.sum(('lat','lon'),skipna=True)/((ds/ds)*coslat).sum(('lat','lon'),skipna=True)
    return ds.weighted(coslat).mean(('lat','lon'),skipna=True)

def linregress(da_y, da_x, dim=None):
    '''xarray-wrapped function of scipy.stats.linregress.
    Note the order of the input arguments x, y is reversed to the original scipy function.'''
    if dim is None:
        dim = [d for d in da_y.dims if d in da_x.dims][0]

    slope, intercept, r, p, stderr = xr.apply_ufunc(_linregress, da_x, da_y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        dask='allowed')
    predicted = da_x * slope + intercept

    slope.attrs['long_name'] = 'slope of the linear regression'
    intercept.attrs['long_name'] = 'intercept of the linear regression'
    r.attrs['long_name'] = 'correlation coefficient'
    p.attrs['long_name'] = 'p-value'
    stderr.attrs['long_name'] = 'standard error of the estimated gradient'
    predicted.attrs['long_name'] = 'predicted values by the linear regression model'

    return xr.Dataset(dict(slope=slope, intercept=intercept,
        r=r, p=p, stderr=stderr, predicted=predicted))

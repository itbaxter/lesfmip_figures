"""
Functions to compute statistics, such as linear trends and p-values, for the data used in the LESFMIP figures.

Author: Tiffany Shaw (tas1@uchicago.edu)
- Ian Baxter (itbaxter@uchicago.edu)

Date:   May 12 2026
"""

# Packages
import xarray as xr
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

def compute_trend_ols(da):
    slope, p_value = xr.apply_ufunc(
        ols, da.time.dt.year, da,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float]
    )

    return xr.Dataset({'slope': slope, 'p_value': p_value})
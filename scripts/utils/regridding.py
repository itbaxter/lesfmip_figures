# %%
import xarray as xr
import numpy as np
import glob as glob

import xesmf as xe

class Regridder:
    def __init__(self, source_ds, target_ds, method='bilinear'):
        self.source_ds = source_ds
        self.target_ds = target_ds
        self.method = method

    def add_lat_lon_bounds(self, ds, lat_name='lat', lon_name='lon'):
        """
        Function to add latitude and longitude bounds to a dataset
        for conservative regridding with xESMF.

        Parameters:
        - ds: xarray Dataset or DataArray containing latitude and longitude coordinates
        - lat_name: Name of the latitude coordinate in the dataset (default is 'lat')
        - lon_name: Name of the longitude coordinate in the dataset (default is 'lon')

        Returns:
        - ds: Dataset with added 'lat_bnds' and 'lon_bnds' variables
        """

        # Get latitude and longitude coordinates
        lat = ds[lat_name]
        lon = ds[lon_name]

        # Calculate latitude bounds
        lat_diff = np.diff(lat) / 2.0
        lat_bnds = np.empty((lat.size, 2), dtype=np.float64)
        lat_bnds[:, 0] = lat - np.concatenate(([lat_diff[0]], lat_diff))
        lat_bnds[:, 1] = lat + np.concatenate((lat_diff, [lat_diff[-1]]))

        # Calculate longitude bounds
        lon_diff = np.diff(lon) / 2.0
        lon_bnds = np.empty((lon.size, 2), dtype=np.float64)
        lon_bnds[:, 0] = lon - np.concatenate(([lon_diff[0]], lon_diff))
        lon_bnds[:, 1] = lon + np.concatenate((lon_diff, [lon_diff[-1]]))

        # Add latitude and longitude bounds to dataset
        ds.coords['lat_bnds'] = (('lat', 'bnds'), lat_bnds)
        ds.coords['lon_bnds'] = (('lon', 'bnds'), lon_bnds)

        return ds

    def regrid(self):
        print('Regridding...')
        
        # Track if input was a DataArray so we can convert back
        is_data_array = isinstance(self.source_ds, xr.DataArray)
        input_name = self.source_ds.name if is_data_array else None
        
        # Convert DataArray to Dataset for regridding
        if is_data_array:
            self.source_ds = self.source_ds.to_dataset()
        
        # Handle coordinate setup for regridding
        if 'nbnd' in list(self.source_ds.coords.keys()):
            self.source_ds['nbnd'] = [1.0, 2.0]
        if 'lat_bnds' not in list(self.source_ds.coords.keys()):
            self.source_ds = self.add_lat_lon_bounds(self.source_ds)
        if len(self.source_ds['lat_bnds'].dims) > 2:
            self.source_ds = self.add_lat_lon_bounds(self.source_ds)
        
        # Perform regridding
        regridder = xe.Regridder(self.source_ds, self.target_ds, self.method, periodic=True)
        dr_out = regridder(self.source_ds, keep_attrs=True)
        
        # If input was DataArray, convert back to DataArray
        if is_data_array and input_name:
            dr_out = dr_out[input_name]
        
        return dr_out

def create_target_grid(GRID_DIR=None, target_lon=np.arange(0, 360, 1), target_lat=np.arange(-90, 90.1, 1)):
    if GRID_DIR is not None:
        files = sorted(glob.glob(f'{GRID_DIR}/*nc'))
        grid = xr.open_dataset(files[0])

        print(list(grid.dims))
        dims = list(grid.dims)
        dim1 = [d for d in dims if 'lat' in d][0]
        dim2 = [d for d in dims if 'lon' in d][0]
        print(dim1,dim2)
        resolution = f'{len(grid[dim1])}x{len(grid[dim2])}'
        print(resolution)

        return xr.Dataset(
            {
                "lat": (["lat"], grid[dim1].data, {"units": "degrees_north"}),
                "lon": (["lon"], grid[dim2].data, {"units": "degrees_east"}),
            }
        )

    else:
        return xr.Dataset(
            {
                "lat": (["lat"], target_lat, {"units": "degrees_north"}),
                "lon": (["lon"], target_lon, {"units": "degrees_east"}),
            }
        )
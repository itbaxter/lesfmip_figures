import xarray as xr
import numpy as np
import xesmf as xe
import glob as glob
import psutil  # For memory profiling
import gc
import os
# %%
def print_memory_usage(stage):
    """Print memory usage at different stages."""
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"[{stage}] Memory usage: {memory_info.rss / (1024 ** 2) / 1000} GB")  # Memory usage in MB



class CMIP6_READER:
    def __init__(self,var,source_id,experiment_id,table_id,ds_out, input_directory, output_directory):
        self.var = var
        self.source_id = source_id
        self.experiment_id = experiment_id
        self.table_id = table_id
        self.ds_out = ds_out
        self.input_directory = input_directory
        self.output_directory = output_directory
        print_memory_usage("Initial")

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

    def regrid(self, ds):
        print('Regridding...')
        
        # Track if input was a DataArray so we can convert back
        is_data_array = isinstance(ds, xr.DataArray)
        input_name = ds.name if is_data_array else None
        
        # Convert DataArray to Dataset for regridding (xESMF works better with Datasets)
        if is_data_array:
            ds = ds.to_dataset()
        
        # Handle coordinate setup for regridding
        if 'nbnd' in list(ds.coords.keys()):
            ds['nbnd'] = [1.0, 2.0]
        if 'lat_bnds' not in list(ds.coords.keys()):
            ds = self.add_lat_lon_bounds(ds)
        if len(ds['lat_bnds'].dims) > 2:
            ds = self.add_lat_lon_bounds(ds)
        
        # Perform regridding
        regridder = xe.Regridder(ds, self.ds_out, "conservative", periodic=True)
        dr_out = regridder(ds, keep_attrs=True)
        
        # If input was DataArray, convert back to DataArray
        if is_data_array and input_name:
            dr_out = dr_out[input_name]
        
        return dr_out

    def get_data(self,member_id):
            files = sorted(glob.glob(f'{self.input_directory}/*/*/*/{self.source_id}/{self.experiment_id}/{member_id}/{self.table_id}/{self.var}/*/*/*nc'))

            ds = xr.open_mfdataset(files,combine='nested',concat_dim='time').drop_duplicates('time')
            if 'plev' in list(ds.coords.keys()):
                ds = ds.assign_coords({'plev': self.ds_out['plev'].values})
            ds.close()
            return ds

    def format(self,ds):
        if 'member_id' not in ds.dims:
            ds = ds.expand_dims('member_id')
            ds['member_id'] = ('member_id', [ds.attrs.get('member_id', 'r1i1p1f1')])

        ds.coords['member_id'] = ('member_id', [f'{self.source_id}_{member.values}' for member in ds["member_id"]])
        
        # Add height as attribute instead of coordinate to avoid dimension issues
        if var == 'tas' or var == 'huss':
            ds.attrs['height'] = 2.0
        elif var == 'uas' or var == 'vas':
            ds.attrs['height'] = 10.0
        if 'source_id' in list(ds.coords.keys()):
            ds = ds.drop_vars('source_id')
        #if 'bnds' not in list(ds.coords.keys()):
        #    ds.coords['bnds'] = [1.0,2.0]
        if 'time_bnds' in list(ds.coords.keys()):
            ds = ds.drop_vars('time_bnds')
        if 'time_bounds' in list(ds.coords.keys()):
            ds = ds.drop_vars('time_bounds')
        if 'dcpp_init_year' in list(ds.coords.keys()):
            ds = ds.drop_vars('dcpp_init_year')
        #if 'plev' in list(ds.coords.keys()):
        #    ds = ds.drop_vars('plev')
        return ds


    def process(self):
        # Process member by member to save memory
        directories = sorted(glob.glob(f'{self.input_directory}/*/*/*/{self.source_id}/{self.experiment_id}/*'))  

        members = [d.split('/')[-1] for d in directories]

        try:
            temp_files = []
            data = []

            for member in members[:]:
                ds_member = self.get_data(member_id=member)

                # Ensure member_id is a dimension
                if 'member_id' not in ds_member.dims:
                    ds_member = ds_member.expand_dims('member_id')
                    ds_member.coords["member_id"] = ('member_id', [f'{ds_member.attrs["source_id"]}_{ds_member.attrs["variant_label"]}'])
                    
                # Only assign lat_bnds and nbnd if they exist in data_vars (not as coordinates)
                coords_to_assign = {}
                if 'lat_bnds' in ds_member.data_vars:
                    coords_to_assign['lat_bnds'] = ds_member['lat_bnds']
                if 'nbnd' in ds_member.data_vars:
                    coords_to_assign['nbnd'] = ds_member['nbnd']
                if coords_to_assign:
                    ds_member = ds_member.assign_coords(coords_to_assign)
                
                print(ds_member[self.var])

                index = self.regrid(ds_member.squeeze()).sortby('time')

                yrstart = index['time.year'][0].values
                yrend = index['time.year'][-1].values

                # Verify time dimension length before creating new coordinate
                time_length = len(index['time']) if 'time' in index.dims else index.dims.get('time', 0)
                time_range = xr.date_range(start=f'{yrstart}-01-01', periods=time_length, freq='MS', use_cftime=True)
                index = index.assign_coords(time=time_range)
                #if self.experiment_id == 'amip':
                index = index.sel(time=slice('1979-01-01','2014-12-31'))

                if index.time.size == 0:
                   continue

                index = self.format(index)

                # Use a sanitized member_id for the filename
                safe_member_id = str(member).replace('/', '_')
                temp_fname = f'{self.output_directory}/temp_{self.var}_{self.source_id}_{self.experiment_id}_{safe_member_id}.nc'
                index.to_netcdf(temp_fname, mode='w')
                print('Written to:',temp_fname)
                temp_files.append(temp_fname)
                data.append(index)
                del ds_member
                del index
                gc.collect()
                print_memory_usage(safe_member_id)

            # Combine member files
            combined_ds = xr.concat(data,dim='member_id')
            del data

            # Get start and end year from combined data for the final filename
            final_yrstart = combined_ds['time.year'].min().values
            final_yrend = combined_ds['time.year'].max().values

            fname = f'{self.output_directory}/{self.var}_CMIP6_{self.source_id}_Amon_{self.experiment_id}_{final_yrstart}-{final_yrend}.nc'

            # Define encoding for compression
            encoding = {v: {'zlib': True, 'complevel': 5} for v in combined_ds.data_vars}

            # Write to netcdf using dask for memory efficiency
            write_job = combined_ds.to_netcdf(
                fname,
                mode='w',
                unlimited_dims=['member_id'],
                compute=False,
                encoding=encoding,
                engine='h5netcdf'
            )
            write_job.compute()

            print('Saved to:',f'{fname}')

            # Clean up temporary files
            #for f in temp_files:
            #    os.remove(f)

            del ds
            del combined_ds

            gc.collect()
        except Exception as e:
            # Print the error message
            print(f"An error occurred: {e}")

    def process(self):
        try:
            files = sorted(glob.glob(f'{self.input_directory}/*/*/*/{self.source_id}/{self.experiment_id}/*/{self.table_id}/{self.var}/*/*/*nc'))
            member_ids = np.unique([f.split('/')[11] for f in files])
            print([m for m in member_ids],len(member_ids))
            if len(member_ids) > 1:
                ds = [self.get_data(member_id) for member_id in member_ids[:]]
                print('Here 2')
                ds = xr.concat(ds,dim='member_id')
                # Clean up individual datasets
                gc.collect()
            else:
                ds = self.get_data(member_ids[0])
            
            # Regrid the full dataset (not just the variable)
            index = self.regrid(ds)
            
            # Extract the variable as DataArray
            if isinstance(index, xr.Dataset):
                index = index[self.var]
            
            index = index.sortby('time')
            yrstart = index['time.year'][0].values
            yrend = index['time.year'][-1].values+1

            print(index)
            # Ensure member_id is a dimension on the DataArray
            if 'member_id' not in index.dims:
                index = index.expand_dims('member_id')

            # Now safely assign member_id coordinate (dimension already exists)
            index.coords['member_id'] = ('member_id', [f'{self.source_id}_{member}' for member in member_ids])
            print(index)

            # Add height as a scalar coordinate (0-D)
            if var == 'tas':
                index.attrs['height'] = 2.0
            elif var in ['uas', 'vas']:
                index.attrs['height'] = 10.0
                
            if 'time_bnds' in list(ds.coords.keys()):
                index = index.drop_vars('time_bnds')
            if 'time_bounds' in list(ds.coords.keys()):
                index = index.drop_vars('time_bounds')
            print_memory_usage("writing out files")
            fname = f'{self.output_directory}/{self.var}_CMIP6_{self.source_id}_{self.table_id}_{self.experiment_id}_{yrstart}-{yrend}.nc'
            index.to_netcdf(fname,mode='w')
            print('Written to:',fname)
            return index
        except Exception as e:
            # Print the error message
            print(f"An error occurred: {e}")


def get_source_ids(var,experiment_id,table_id,input_directory,source_id='None'):
    if source_id == 'None':
        files = sorted(glob.glob(f'{input_directory}/*/*/*/*/{experiment_id}/*/{table_id}/{var}/*/*/*nc'))
    else:
        files = sorted(glob.glob(f'{input_directory}/*/*/*/{source_id}/{experiment_id}/*/{table_id}/{var}/*/*/*nc'))

    source_ids = []
    member_ids = []
    for f in files:
        source_ids.append(f.split('/')[9])
        member_ids.append(f.split('/')[11])
    print(files[0])
    print(source_ids,member_ids)

    return np.unique(source_ids),np.unique(member_ids)

if __name__=="__main__":
    #=========================================================================
    # Make changes here
    #=========================================================================
    # Experiment you want to regrid and standardize
    EXPERIMENT_ID = 'hist-aer'
    # Frequency of outputs Amon, day
    TABLE_ID = 'Amon'
    # Variable to regrid
    DATA_VARS = ['ua']
    # File with grid template. Could manually make the trend below
    GRID_DIR = '/project/tas1/itbaxter/for-tiffany/amip/180x360/ta/Amon/'
    # Input directory for ESGF files. Should be organized as */source_id/experiment_id/member_id/table_id/var/*nc
    # NOTE: This points to your .esgf directory, so assumes you downloaded the data using get_cmip6-esgf2.py. Use process_local.py if you have the data organized differently on your system.
    INPUT_DIR = '/project/tas1/itbaxter/for-tiffany/.esgf/'

    # Output directory for regridded files. Will be organized by experiment_id/resolution/var/*source_id*nc
    OUTPUT_DIR = '/project/tas1/itbaxter/for-tiffany/'

    #=========================================================================
    # Shouldn't need to change these but you could
    #=========================================================================
    files = sorted(glob.glob(f'{GRID_DIR}/*nc'))
    grid = xr.open_dataset(files[0])

    print(list(grid.dims))
    dims = list(grid.dims)
    dim1 = [d for d in dims if 'lat' in d][0]
    dim2 = [d for d in dims if 'lon' in d][0]
    print(dim1,dim2)
    resolution = f'{len(grid[dim1])}x{len(grid[dim2])}'
    print(resolution)

    # %%
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], grid[dim1].data, {"units": "degrees_north"}),
            "lon": (["lon"], grid[dim2].data, {"units": "degrees_east"}),
            "plev": (["plev"], grid['plev'].data, {"units": "Pa"}), 
        }
    )

    for var in DATA_VARS:
        source_ids,_ = get_source_ids(var,EXPERIMENT_ID,TABLE_ID,INPUT_DIR)
        output_directory = f'{OUTPUT_DIR}/{EXPERIMENT_ID}/{resolution}/{var}/'
        print(output_directory)

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        out_files = sorted(glob.glob(f'{output_directory}/*nc'))
        finished = [f.split('/')[-1].split('_')[2] for f in out_files]
        #finished = []
        #source_ids = ['CESM2']
        print(source_ids)
        print('Finished:',finished)

        for source_id in source_ids[:]:
            if source_id not in finished:
                print(source_id)
                CMIP6_READER(var,source_id,EXPERIMENT_ID,TABLE_ID,ds_out,output_directory).process()

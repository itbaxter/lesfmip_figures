# %%
import os
import numpy as np
import xarray as xr
import xesmf as xe
import gc
import pandas as pd
import glob as glob
import warnings
import logging
from datetime import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import intake_esgf

# Set dask chunk size to manage memory better
xr.set_options(keep_attrs=True)
import dask
dask.config.set({'array.chunk-size': '128MiB'})

# Import intake-esgf2
import intake_esgf

print("Pandas version:", pd.__version__)
print("Intake-ESGF version:", intake_esgf.__version__)

# %%
import psutil  # For memory profiling
def print_memory_usage(stage):
    """Print memory usage at different stages."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 ** 2) / 1000
    print(f"[{stage}] Memory usage: {memory_gb:.2f} GB")
    
    # Warning if memory usage is getting high
    if memory_gb > 6.0:  # Warning at 6GB
        print(f"WARNING: High memory usage detected ({memory_gb:.2f} GB)")
        gc.collect()  # Force garbage collection
        memory_info_after = psutil.Process().memory_info()
        memory_gb_after = memory_info_after.rss / (1024 ** 2) / 1000
        print(f"After garbage collection: {memory_gb_after:.2f} GB")
    
    return memory_gb

# %%
def setup_logger(output_directory, log_filename='cmip6_processing.log'):
    """
    Set up logging to both file and console.
    
    Parameters:
    -----------
    output_directory : str
        Directory where log file will be saved
    log_filename : str
        Name of the log file
        
    Returns:
    --------
    logger : logging.Logger
        Configured logger instance
    """
    # Ensure the output directory exists before creating handlers.
    os.makedirs(output_directory, exist_ok=True)

    # Create logger
    logger = logging.getLogger('CMIP6_Processing')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create file handler
    log_path = os.path.join(output_directory, log_filename)
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# %%
class CMIP6_ESGF_READER:
    def __init__(self, var, source_id, experiment_id, ds_out, output_directory,
                 table_id='Amon', grid_label='gn', member_id='r1i1p1f1', logger=None):
        self.var = var
        self.source_id = source_id
        self.experiment_id = experiment_id
        self.table_id = table_id
        self.grid_label = grid_label
        self.member_id = member_id
        self.ds_out = ds_out
        self.output_directory = output_directory
        self.logger = logger or logging.getLogger('CMIP6_Processing')
        self.print_memory_usage("Initial")

        # Initialize intake-esgf2 catalog
        self.cat = intake_esgf.ESGFCatalog()
        try:
            self.local_cache = intake_esgf.conf.get("local_cache")
            if '~/' in self.local_cache:
                print('Putting data in home directory!!!!!!')
                exit() 
            # Ensure local_cache is a string, not a list
            #if isinstance(self.local_cache, list):
            #    self.local_cache = self.local_cache[0] if self.local_cache else None
            print(f"Local cache for ESGF: {self.local_cache}")
        except Exception:
            print("Local cache is None")
            self.local_cache = None

    def print_memory_usage(self, stage):
        """Print memory usage at different stages."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024 ** 2) / 1000
        print(f"[{stage}] Memory usage: {memory_gb:.2f} GB")
        
        # Warning if memory usage is getting high (lowered to 6GB for preemptive action)
        if memory_gb > 7.5:  # Warning at 6GB to trigger preemptive splitting
            print(f"WARNING: High memory usage detected ({memory_gb:.2f} GB)")
            gc.collect()  # Force garbage collection
            memory_info_after = psutil.Process().memory_info()
            memory_gb_after = memory_info_after.rss / (1024 ** 2) / 1000
            print(f"After garbage collection: {memory_gb_after:.2f} GB")
            self.high_memory_warning = True  # Set warning flag
        else:
            self.high_memory_warning = False  # Reset warning flag if memory is under control
        
        return memory_gb

    def search_and_download(self):
        """Search for data using intake-esgf2 and download if needed."""
        try:
            # Search for the data
            search_results = self.cat.search(
                variable_id=self.var,
                source_id=self.source_id,
                experiment_id=self.experiment_id,
                table_id=self.table_id,
                grid_label=self.grid_label,
                member_id=self.member_id
            )

            # Get the first result (or you can filter further)
            result = search_results.iloc[0]

            # Download the data
            ds = result.to_dask()

            self.print_memory_usage("Data loaded")
            return ds

        except Exception as e:
            print(f"Error in search_and_download: {e}")
            return None
  
    def get_data_opendap(self):
        """Alternative method using OpenDAP URLs from ESGF."""
        try:
            # Search for data
            search_results = self.cat.search(
                variable_id=self.var,
                source_id=self.source_id,
                experiment_id=self.experiment_id,
                table_id=self.table_id
            )

            # Get OpenDAP URLs
            urls = search_results.get_download_urls()
            opendap_urls = [url for url in urls if 'opendap' in url or '.nc' in url]

            if not opendap_urls:
                raise ValueError("No OpenDAP URLs found")

            # Open dataset from OpenDAP
            ds = xr.open_mfdataset(opendap_urls, combine='by_coords',
                                 data_vars='minimal', coords='minimal', compat='override')

            self.print_memory_usage("Data loaded via OpenDAP")
            return ds

        except Exception as e:
            print(f"Error in get_data_opendap: {e}")
            return None

    def add_labels(self, ds, member_id):
        """Add member_id labels to the dataset."""
        # Add member_id if not present
        if 'member_id' not in ds.dims:
            ds = ds.expand_dims('member_id')
            ds['member_id'] = ('member_id', [member_id])
        
        # Update member_id coordinate
        ds.coords['member_id'] = ('member_id', [f'{self.source_id}_{member_id}'])
        
        return ds

    def process(self):
        """Main processing function using the specified approach."""
        processing_start = datetime.now()
        variant_results = []  # Track results for each variant
        
        try:
            self.logger.info(f"=" * 80)
            self.logger.info(f"Starting processing: {self.source_id}")
            self.logger.info(f"Variable: {self.var}, Experiment: {self.experiment_id}, Table: {self.table_id}")
            self.print_memory_usage(f"Starting {self.source_id}")
            
            # Search for the data first
            cat_filtered = self.cat.search(
                variable_id=self.var,
                source_id=self.source_id,
                experiment_id=self.experiment_id,
                table_id=self.table_id,
                grid_label=self.grid_label,
                #member_id='r1i1p1f1',
            )
            
            # Get filtered catalog
            member_ids = cat_filtered.df['member_id'].unique()
            
            # Limit member_ids to avoid memory issues - process only first 3
            if len(member_ids) > 1:
                member_ids = member_ids[:10]
                #print(f"Limiting to first individual member_ids to manage memory: {member_ids}")
            
            self.logger.info(f"Found {len(member_ids)} member_ids: {list(member_ids)}")
            
            # Convert to dataset dictionary with memory management
            self.print_memory_usage("Before loading datasets")
            try:
                # Try with on_error='skip' to skip failed datasets if supported
                try:
                    dsd = cat_filtered.to_dataset_dict(add_measures=False, on_error='skip')
                except TypeError:
                    # If on_error parameter not supported, try without it
                    dsd = cat_filtered.to_dataset_dict(add_measures=False)
            except Exception as e:
                # If batch loading fails completely, try to load each member_id separately
                self.logger.warning(f"Batch loading failed: {e}")
                self.logger.info("Attempting to load member_ids individually...")
                dsd = {}
                
                for member_id in member_ids:
                    try:
                        # Search for this specific member_id
                        member_cat = self.cat.search(
                            variable_id=self.var,
                            source_id=self.source_id,
                            experiment_id=self.experiment_id,
                            table_id=self.table_id,
                            grid_label=self.grid_label,
                            member_id=member_id
                        )
                        
                        # Try to load this member's dataset
                        member_dsd = member_cat.to_dataset_dict(add_measures=False)
                        dsd.update(member_dsd)
                        self.logger.info(f"Successfully loaded {member_id}")
                    except Exception as member_error:
                        self.logger.error(f"Failed to load {member_id}: {member_error}")
                        continue
                
                if not dsd:
                    self.logger.error(f"Failed to load any datasets for {self.source_id}")
                    return None
                else:
                    self.logger.info(f"Successfully loaded {len(dsd)} variants individually")
            
            self.print_memory_usage("After loading datasets")

            variant_labels = list(dsd.keys())
            variant_labels_max = variant_labels
            batch_files = []  # Track intermediate batch files
            
            self.logger.info(f"Processing {len(variant_labels_max)} variant labels: {variant_labels_max}")
            print(f"{self.source_id}, member_ids: {member_ids}, count: {len(member_ids)}")
            print(f"Variant labels: {variant_labels_max}")

            # Log final summary
            processing_duration = (datetime.now() - processing_start).total_seconds()
            self.logger.info(f"=" * 80)
            self.logger.info(f"COMPLETED: {self.source_id}")
            self.logger.info(f"Total processing time: {processing_duration:.1f} seconds ({processing_duration/60:.1f} minutes)")
            self.logger.info(f"Files created: {len(batch_files) if batch_files else 1}")
            if batch_files:
                self.logger.info(f"Batch files: {batch_files}")
            self.logger.info(f"=" * 80)
           
            return True  # Return success indicator instead of data
        except Exception as e:
            # Log the exception
            error_msg = f"FATAL ERROR processing {self.source_id}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            
            print(f"An error occurred processing {self.source_id}: {e}")
            # Force cleanup on error
            gc.collect()
            return None

# %%
def setup_target_grid(template_file=None, lat_res=1.0, lon_res=1.0):
    """Set up target grid for regridding."""
    if template_file and os.path.exists(template_file):
        # Use existing file as template
        template = xr.open_dataset(template_file)
        dims = list(template.dims)
        lat_dim = [d for d in dims if 'lat' in d][0]
        lon_dim = [d for d in dims if 'lon' in d][0]

        ds_out = xr.Dataset({
            "lat": (["lat"], template[lat_dim].data, {"units": "degrees_north"}),
            "lon": (["lon"], template[lon_dim].data, {"units": "degrees_east"}),
        })
    else:
        # Create regular lat/lon grid
        lat = np.arange(-89.5, 89.5 + lat_res, lat_res)
        lon = np.arange(0, 360, lon_res)

        ds_out = xr.Dataset({
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "lon": (["lon"], lon, {"units": "degrees_east"}),
        })

    return ds_out

def check_available_models(variables, experiment_ids, table_id='Amon', grid_label='gn'):
    """
    Check what source_ids (models) are available for given variables and experiments.
    
    Parameters:
    -----------
    variables : list
        List of variable names to check
    experiment_ids : list
        List of experiment IDs to check
    table_id : str
        Table ID (default: 'Amon')
    grid_label : str
        Grid label (default: 'gn')
        
    Returns:
    --------
    dict
        Dictionary with structure: {experiment_id: {variable: [list_of_source_ids]}}
    """
    
    # Initialize intake-esgf2 catalog
    cat = intake_esgf.ESGFCatalog()
    search_results = cat.search(
                    variable_id=variables[0],
                    experiment_id=experiment_ids[0],
                    table_id=table_id,
                    grid_label=grid_label
                )
    print(search_results)
    source_ids = search_results.df['source_id'].unique().tolist()
    print(source_ids)
    
    available_models = {}
    
    for experiment_id in experiment_ids:
        available_models[experiment_id] = {}
        print(f"\nChecking experiment: {experiment_id}")
        
        for var in variables:
            print(f"  Checking variable: {var}")
            
            try:
                # Search for available data
                search_results = cat.search(
                    variable_id=var,
                    experiment_id=experiment_id,
                    table_id=table_id,
                    grid_label=grid_label
                )
                
                if len(search_results.df['source_id']) > 0:
                    # Get unique source_ids
                    source_ids = search_results.df['source_id'].unique().tolist()
                    available_models[experiment_id][var] = sorted(source_ids)
                    print(f"    Found {len(source_ids)} models: {source_ids[:3]}{'...' if len(source_ids) > 3 else ''}")
                else:
                    available_models[experiment_id][var] = []
                    print(f"    No models found for {var}")
                    
            except Exception as e:
                print(f"    Error searching for {var}: {e}")
                available_models[experiment_id][var] = []
    
    return available_models

def get_common_models(available_models, variables, experiment_ids, min_experiments=1):
    """
    Get models that are available for all specified variables and a minimum number of experiments.
    
    Parameters:
    -----------
    available_models : dict
        Output from check_available_models()
    variables : list
        List of variables to check
    experiment_ids : list
        List of experiments to check
    min_experiments : int
        Minimum number of experiments a model must be available for
        
    Returns:
    --------
    list
        List of source_ids that meet the criteria
    """
    
    model_counts = {}
    
    for experiment_id in experiment_ids:
        if experiment_id in available_models:
            # Get models that have ALL variables for this experiment
            experiment_models = None
            
            for var in variables:
                var_models = set(available_models[experiment_id].get(var, []))
                if experiment_models is None:
                    experiment_models = var_models
                else:
                    experiment_models = experiment_models.intersection(var_models)
            
            # Count how many experiments each model appears in
            for model in experiment_models:
                model_counts[model] = model_counts.get(model, 0) + 1
    
    # Filter models that appear in at least min_experiments
    common_models = [model for model, count in model_counts.items() 
                    if count >= min_experiments]
    
    return sorted(common_models)

def print_availability_summary(available_models, variables, experiment_ids):
    """Print a summary of model availability."""
    
    print("\n" + "="*60)
    print("MODEL AVAILABILITY SUMMARY")
    print("="*60)
    
    for experiment_id in experiment_ids:
        if experiment_id in available_models:
            print(f"\nExperiment: {experiment_id}")
            print("-" * 40)
            
            for var in variables:
                models = available_models[experiment_id].get(var, [])
                print(f"  {var:10s}: {len(models):3d} models")
                if len(models) > 0:
                    print(f"             {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
    
    # Show common models across all variables and experiments
    common_models = get_common_models(available_models, variables, experiment_ids, 
                                    min_experiments=len(experiment_ids))
    
    print(f"\nModels with ALL variables in ALL experiments ({len(common_models)}):")
    print("-" * 50)
    for model in common_models:
        print(f"  {model}")
    
    # Show models with all variables in at least one experiment
    any_experiment_models = get_common_models(available_models, variables, experiment_ids, 
                                            min_experiments=1)
    
    print(f"\nModels with ALL variables in at least ONE experiment ({len(any_experiment_models)}):")
    print("-" * 50)
    for model in any_experiment_models:
        print(f"  {model}")

def batch_process_cmip6_with_availability_check(variables, experiment_ids, output_base_dir, 
                                               target_grid=None, table_id='Amon', 
                                               grid_label='gn', 
                                               min_experiments=1,
                                               specific_models=None):
    """
    Batch process CMIP6 datasets with automatic availability checking.
    
    Parameters:
    -----------
    variables : list
        List of variable names to process
    experiment_ids : list
        List of experiment IDs to process
    output_base_dir : str
        Base directory for output files
    target_grid : xarray.Dataset, optional
        Target grid for regridding
    table_id : str
        Table ID (default: 'Amon')
    grid_label : str
        Grid label (default: 'gn')
    min_experiments : int
        Minimum number of experiments a model must be available for
    specific_models : list, optional
        If provided, only process these specific models (still checks availability)
    """
    
    print("Checking model availability...")
    
    # Check what models are available
    available_models = check_available_models(variables, experiment_ids, table_id, grid_label)
    
    # Print summary
    print_availability_summary(available_models, variables, experiment_ids)
    
    # Get models to process
    if specific_models is not None:
        # Filter specific models by availability
        common_models = get_common_models(available_models, variables, experiment_ids, 
                                        min_experiments=min_experiments)
        source_ids = [model for model in specific_models if model in common_models]
        print(f"\nFiltered specific models by availability: {len(source_ids)} models")
    else:
        # Use all available models
        source_ids = get_common_models(available_models, variables, experiment_ids, 
                                     min_experiments=min_experiments)
        print(f"\nUsing all available models: {len(source_ids)} models")
    
    if len(source_ids) == 0:
        print("No models meet the criteria. Exiting.")
        return
    
    print(f"Models to process: {source_ids}")
    
    # Set up target grid
    ds_out = setup_target_grid(target_grid)
    
    resolution = f"{len(ds_out.lat)}x{len(ds_out.lon)}"
    print(f"Target grid: {resolution}")
    # Process each experiment and variable
    for experiment_id in experiment_ids:
        for var in variables:
            print(f"\nProcessing {var} for experiment {experiment_id}")
            
            # Filter models available for this specific variable and experiment
            var_models = available_models[experiment_id].get(var, [])
            process_models = [model for model in source_ids if model in var_models]
            
            if len(process_models) == 0:
                print(f"No models available for {var} in {experiment_id}")
                continue
            
            print(f"Processing {len(process_models)} models for {var}")
            
            # Create output directory
            output_directory = f'{output_base_dir}/.logs'
            
            # Set up logger for this experiment/variable combination
            log_filename = f'cmip6_processing_{experiment_id}_{var}_{table_id}.log'
            logger = setup_logger(output_directory, log_filename)
            logger.info(f"=" * 80)
            logger.info(f"Starting batch processing")
            logger.info(f"Experiment: {experiment_id}, Variable: {var}, Table: {table_id}")
            logger.info(f"Output directory: {output_directory}")
            logger.info(f"Target grid: {resolution}")
            logger.info(f"=" * 80)
            
            # Check for existing files
            existing_files = sorted(glob.glob(f'{output_directory}/*nc'))
            finished_models = [f.split('_')[2] for f in existing_files]
            
            remaining_models = [s for s in process_models if s not in finished_models]
            logger.info(f"Total models to process: {len(process_models)}")
            logger.info(f"Already completed: {len(finished_models)}")
            logger.info(f"Remaining models: {len(remaining_models)}")
            if finished_models:
                logger.info(f"Completed models: {finished_models}")
            print(f"Remaining models to process: {len(remaining_models)}")
            
            # Process each model
            for i, source_id in enumerate(remaining_models):
                print(f"\nProcessing {source_id} ({i+1}/{len(remaining_models)})...")
                logger.info(f"\n{'='*80}")
                logger.info(f"Processing model {i+1}/{len(remaining_models)}: {source_id}")
                print_memory_usage(f"Before {source_id}")
                
                reader = CMIP6_ESGF_READER(
                    var=var,
                    source_id=source_id,
                    experiment_id=experiment_id,
                    ds_out=ds_out,
                    output_directory=output_directory,
                    table_id=table_id,
                    grid_label=grid_label,
                    logger=logger
                )
                
                result = reader.process()
                
                # Clean up reader and force garbage collection
                del reader
                gc.collect()
                
                if result is not None:
                    logger.info(f"Successfully processed {source_id}")
                    print(f"Successfully processed {source_id}")
                else:
                    logger.error(f"Failed to process {source_id}")
                    print(f"Failed to process {source_id}")
                
                print_memory_usage(f"After {source_id}")
                
                # Additional cleanup between models
                if i % 2 == 1:  # Every 2 models, force more aggressive cleanup
                    print("Performing intermediate cleanup...")
                    gc.collect()
                    print_memory_usage("After intermediate cleanup")
            
            # Log final summary for this experiment/variable
            logger.info(f"\n{'='*80}")
            logger.info(f"BATCH PROCESSING COMPLETED")
            logger.info(f"Experiment: {experiment_id}, Variable: {var}")
            logger.info(f"Total models processed: {len(remaining_models)}")
            logger.info(f"{'='*80}\n")

if __name__ == "__main__":
    #===========================================================================
    # Modify these parameters as needed for your specific use case 
    #===========================================================================
    TARGET_GRID = glob.glob('./hist-aer/71x144/ua/*nc')[0]

    # Define variables and experiments
    DATA_VARS = ['ua'] #,'pr','huss']
    EXPERIMENT_IDS = ['hist-GHG', 'hist-aer', 'hist-nat', 'hist-stratO3','historical','ssp585'] #,'ssp585','amip']
    TABLE_ID = 'Amon'
    OUTPUT_BASE_DIR = '/project/tas1/itbaxter/for-tiffany/IPCC_figures/lesfmip_figures/raw_data/'

    #===========================================================================
    # Can pick Option depending on what you want. 
    #===========================================================================

    # Best to change in the ~/.config/intake-esgf/conf.yaml
    os.environ['ESGF_PERSISTENT_CACHE'] = f'{OUTPUT_BASE_DIR}/.esgf'

    # Option 1: Check availability and use all available models
    #batch_process_cmip6_with_availability_check(
    #    variables=DATA_VARS,
    #    experiment_ids=EXPERIMENT_IDS,
    #    tabled_ids=table_ids,
    #    output_base_dir=OUTPUT_BASE_DIR,
    #    min_experiments=2  # Require models to be in both experiments
    #)
    
    # Option 2: Check availability but only process specific models
    preferred_models = ['ACCESS-ESM1-5'] #,'CESM2','GISS-E2-1-G', 'CanESM5','ACCESS-CM2''ACCESS-ESM1-5','FGOALS-g3','HadGEM3-GC31-LL','MIROC6','MRI-ESM2-0','NorESM2-LM']
    batch_process_cmip6_with_availability_check(
        variables=DATA_VARS,
        experiment_ids=EXPERIMENT_IDS,
        table_id=TABLE_ID,
        target_grid=TARGET_GRID,
        output_base_dir=OUTPUT_BASE_DIR,
        specific_models=preferred_models,
        min_experiments=1  # Allow models from at least one experiment
    )
    
    # Option 3: Just check availability without processing
    #available = check_available_models(DATA_VARS, EXPERIMENT_IDS)
    #print_availability_summary(available, DATA_VARS, EXPERIMENT_IDS)


import cdsapi
import numpy as np
import os

years = np.arange(1979,2024,1)
data_var = "u_component_of_wind"

dataset = "reanalysis-era5-pressure-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": data_var,
    "pressure_level": [
        "250", 
        "850",
    ],
    "year": [
        f"{year}" for year in years
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download(f"./Reanalysis/era5_monthly_{data_var}_{years[0]}-{years[-1]}.nc")

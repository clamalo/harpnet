import xarray as xr
import os
import matplotlib.pyplot as plt
from calendar import monthrange
import cartopy.crs as ccrs
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings("ignore")

def generate_year_month_range(start_year_month, end_year_month):
    start_year, start_month = start_year_month
    end_year, end_month = end_year_month
    
    current_year, current_month = start_year, start_month
    year_month_pairs = []
    
    while (current_year, current_month) <= (end_year, end_month):
        year_month_pairs.append((current_year, current_month))
        if current_month == 12:
            current_year += 1
            current_month = 1
        else:
            current_month += 1
            
    return year_month_pairs



for year,month in tqdm(generate_year_month_range((1981,1), (1984,12))):

    files = []
    num_days = monthrange(year, month)[1]
    for day in range(1, num_days + 1):
        for hour in range(0, 24, 3):
            if (year, month, day, hour) == (1979, 10, 1, 0) or (year, month, day, hour) == (1979, 10, 1, 1) or (year, month, day, hour) == (1979, 10, 1, 2):
                continue
            files.append(f'/Volumes/T9/summed/{year}-{month:02d}-{day:02d}_{hour:02d}.nc')

    month_ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')

    month_ds = month_ds.sortby('time')
    month_ds.to_netcdf(f'/Volumes/T9/monthly/{year}-{month:02d}.nc')
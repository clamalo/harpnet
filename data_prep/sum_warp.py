import xarray as xr
import os
import numpy as np
from calendar import monthrange
import dask
from dask.diagnostics import ProgressBar
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
warnings.filterwarnings("ignore")
import glob
import concurrent.futures


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

@dask.delayed
def process_hourly_data(year, month, day, hour):
    print(f'Processing {year}-{month:02d}-{day:02d} {hour:02d}:00:00')

    current_datetime = datetime(year, month, day, hour)
    prior_datetime = current_datetime - timedelta(hours=1)
    prior_2_datetime = current_datetime - timedelta(hours=2)

    ds = xr.open_dataset(f'/Volumes/T9/nc/PREC_ACC_NC.wrf2d_d01_{current_datetime.year}-{current_datetime.month:02d}-{current_datetime.day:02d}_{current_datetime.hour:02d}:00:00.nc')
    prior_ds = xr.open_dataset(f'/Volumes/T9/nc/PREC_ACC_NC.wrf2d_d01_{prior_datetime.year}-{prior_datetime.month:02d}-{prior_datetime.day:02d}_{prior_datetime.hour:02d}:00:00.nc')
    prior_2_ds = xr.open_dataset(f'/Volumes/T9/nc/PREC_ACC_NC.wrf2d_d01_{prior_2_datetime.year}-{prior_2_datetime.month:02d}-{prior_2_datetime.day:02d}_{prior_2_datetime.hour:02d}:00:00.nc')
    datasets = [ds, prior_ds, prior_2_ds]
    ds = xr.concat(datasets, dim='Time')
    ds = ds.sum(dim='Time')
    ds.to_netcdf(f'/Volumes/T9/summed/{year}-{month:02d}-{day:02d}_{hour:02d}_.nc')

    os.system(f"gdalwarp -t_srs EPSG:4326 /Volumes/T9/summed/{year}-{month:02d}-{day:02d}_{hour:02d}_.nc /Volumes/T9/summed/{year}-{month:02d}-{day:02d}_{hour:02d}_new.nc > /dev/null 2>&1")
    os.system(f"rm /Volumes/T9/summed/{year}-{month:02d}-{day:02d}_{hour:02d}_.nc")

    ds = xr.open_dataset(f'/Volumes/T9/summed/{year}-{month:02d}-{day:02d}_{hour:02d}_new.nc')
    ds = ds.drop_vars('crs').rename({'Band1': 'tp'})
    new_time = np.array([np.datetime64(f'{year}-{month:02d}-{day:02d}T{hour:02d}:00:00')])
    ds = ds.assign_coords(time=new_time)
    ds.to_netcdf(f'/Volumes/T9/summed/{year}-{month:02d}-{day:02d}_{hour:02d}.nc')
    os.system(f"rm /Volumes/T9/summed/{year}-{month:02d}-{day:02d}_{hour:02d}_new.nc")


def delete_files_except_last_two_of_month(year, month):
    files = glob.glob(f'/Volumes/T9/nc/PREC_ACC_NC.wrf2d_d01_{year}-{month:02d}*.nc')
    files_to_delete = sorted(files)[:-2]  # Preserving the last two files of the month
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(os.remove, files_to_delete)


def process_month(year, month):
    days_in_month = monthrange(year, month)[1]
    tasks = []

    for day in range(1, days_in_month + 1):
        for hour in range(0, 24, 1):
            if (year, month, day, hour) == (1979, 10, 1, 0) or (year, month, day, hour) == (1979, 10, 1, 1) or (year, month, day, hour) == (1979, 10, 1, 2):
                continue
            tasks.append((year, month, day, hour))
    
    #sort the tasks by date
    tasks = sorted(tasks, key=lambda x: datetime(x[0], x[1], x[2], x[3]))

    tasks = [process_hourly_data(year, month, day, hour) for year, month, day, hour in tasks]
    dask.compute(*tasks)

if __name__ == "__main__":
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)

    year_month_pairs = generate_year_month_range((1985, 12), (2022, 9))

    for year, month in year_month_pairs:
        process_month(year, month)
        delete_files_except_last_two_of_month(year, month)

    client.close()
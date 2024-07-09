import sys, os
import requests
import tarfile
import xarray as xr
import os
import numpy as np
from calendar import monthrange
import dask
from dask.diagnostics import ProgressBar
import warnings
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
warnings.filterwarnings("ignore")
from tqdm import tqdm
import re
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


def create_dirs(tar_dir, raw_dir, summed_dir, monthly_dir):
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if not os.path.exists(summed_dir):
        os.makedirs(summed_dir)
    if not os.path.exists(monthly_dir):
        os.makedirs(monthly_dir)


def check_file_status(filepath):
    sys.stdout.write('\r')
    sys.stdout.flush()
    size_mb = int(os.stat(filepath).st_size)/ 1000**2
    sys.stdout.write('Downloaded %.1f MB' % (size_mb,))
    sys.stdout.flush()


def download_and_untar(year,month,tar_dir,raw_dir):
    with open('/Volumes/T9/filelist.txt', 'r') as filehandle:
        filelist = [line.strip() for line in filehandle.readlines()]

        dspath = 'https://request.rda.ucar.edu/dsrqst/MALOTT748122/'
        for file in filelist:
            filename=dspath+file

            year_month = str(year) + '-' + str(month).zfill(2)
            if year_month not in file:
                continue

            file_base = os.path.basename(file)

            # get rid of the first part when you do .split('.')
            file_base = file_base.split('.')[1:]
            file_base = '.'.join(file_base)

            print('Downloading', file_base)

            req = requests.get(filename, allow_redirects=True, stream=True)
            with open(os.path.join(tar_dir, file_base), 'wb') as outfile:
                chunk_size = 10485760
                for chunk in req.iter_content(chunk_size=chunk_size):
                    outfile.write(chunk)
                    check_file_status(os.path.join(tar_dir, file_base))

            with tarfile.open(os.path.join(tar_dir, file_base)) as tar:
                for member in tqdm(tar.getmembers()):
                    member.name = member.name.split('.')[1:]
                    member.name = '.'.join(member.name)
                    tar.extract(member, path=raw_dir)

            os.remove(os.path.join(tar_dir, file_base))
            print('Extracted', file_base)
            print()


@dask.delayed
def process_hourly_data(year, month, day, hour, raw_dir, summed_dir):

    print(f'Processing {year}-{month:02d}-{day:02d} {hour:02d}:00:00')

    current_datetime = datetime(year, month, day, hour)
    prior_datetime = current_datetime - timedelta(hours=1)
    prior_2_datetime = current_datetime - timedelta(hours=2)

    ds = xr.open_dataset(f'{raw_dir}PREC_ACC_NC.wrf2d_d01_{current_datetime.year}-{current_datetime.month:02d}-{current_datetime.day:02d}_{current_datetime.hour:02d}:00:00.nc')
    prior_ds = xr.open_dataset(f'{raw_dir}PREC_ACC_NC.wrf2d_d01_{prior_datetime.year}-{prior_datetime.month:02d}-{prior_datetime.day:02d}_{prior_datetime.hour:02d}:00:00.nc')
    prior_2_ds = xr.open_dataset(f'{raw_dir}PREC_ACC_NC.wrf2d_d01_{prior_2_datetime.year}-{prior_2_datetime.month:02d}-{prior_2_datetime.day:02d}_{prior_2_datetime.hour:02d}:00:00.nc')
    datasets = [ds, prior_ds, prior_2_ds]
    ds = xr.concat(datasets, dim='Time')
    ds = ds.sum(dim='Time')
    ds.to_netcdf(f'{summed_dir}{year}-{month:02d}-{day:02d}_{hour:02d}_.nc')

    os.system(f"gdalwarp -t_srs EPSG:4326 {summed_dir}{year}-{month:02d}-{day:02d}_{hour:02d}_.nc {summed_dir}{year}-{month:02d}-{day:02d}_{hour:02d}_new.nc > /dev/null 2>&1")
    os.system(f"rm {summed_dir}{year}-{month:02d}-{day:02d}_{hour:02d}_.nc")

    ds = xr.open_dataset(f'{summed_dir}{year}-{month:02d}-{day:02d}_{hour:02d}_new.nc')
    ds = ds.drop_vars('crs').rename({'Band1': 'tp'})
    new_time = np.array([np.datetime64(f'{year}-{month:02d}-{day:02d}T{hour:02d}:00:00')])
    ds = ds.assign_coords(time=new_time)
    ds.to_netcdf(f'{summed_dir}{year}-{month:02d}-{day:02d}_{hour:02d}.nc')
    os.system(f"rm {summed_dir}{year}-{month:02d}-{day:02d}_{hour:02d}_new.nc")


def process_month(year, month, raw_dir, summed_dir):
    days_in_month = monthrange(year, month)[1]
    tasks = []
    for day in range(1, days_in_month + 1):
        for hour in range(0, 24, 1):
            if (year, month, day, hour) == (1979, 10, 1, 0) or (year, month, day, hour) == (1979, 10, 1, 1) or (year, month, day, hour) == (1979, 10, 1, 2):
                continue
            tasks.append((year, month, day, hour, raw_dir, summed_dir))
    #sort the tasks by date
    tasks = sorted(tasks, key=lambda x: datetime(x[0], x[1], x[2], x[3]))
    tasks = [process_hourly_data(year, month, day, hour, raw_dir, summed_dir) for year, month, day, hour, raw_dir, summed_dir in tasks]
    with ProgressBar():
        dask.compute(*tasks)


def monthly_dataset(year,month,summed_dir,montly_dir):

    hourly_files = []
    num_days = monthrange(year, month)[1]
    for day in range(1, num_days + 1):
        for hour in range(0, 24, 1):
            if (year, month, day, hour) == (1979, 10, 1, 0) or (year, month, day, hour) == (1979, 10, 1, 1) or (year, month, day, hour) == (1979, 10, 1, 2):
                continue
            hourly_files.append(f'{summed_dir}{year}-{month:02d}-{day:02d}_{hour:02d}.nc')
    month_ds = xr.open_mfdataset(hourly_files, combine='nested', concat_dim='time')
    print(month_ds)
    month_ds = month_ds.sortby('time')
    month_ds.to_netcdf(f'{montly_dir}{year}-{month:02d}.nc')





if __name__ == "__main__":

    tar_dir = '/Users/clamalo/documents/harpnet/data_prep/tar/'
    raw_dir = '/Users/clamalo/documents/harpnet/data_prep/raw/'
    summed_dir = '/Users/clamalo/documents/harpnet/data_prep/summed/'
    monthly_dir = '/Users/clamalo/documents/harpnet/data_prep/monthly/'

    create_dirs(tar_dir, raw_dir, summed_dir, monthly_dir)

    for year in range(1979,2023):
        start_month = 1 if year != 1979 else 10
        end_month = 12 if year != 2022 else 9
        for month in range(start_month, end_month + 1):

            print('Downloading & untarring')
            download_and_untar(year, month, tar_dir, raw_dir)


            print('Summing & warping')
            cluster = LocalCluster(n_workers=10, threads_per_worker=1)
            client = Client(cluster)
            process_month(year, month, raw_dir, summed_dir)
            client.close()
            cluster.close()


            print('Creating monthly dataset')
            monthly_dataset(year, month, summed_dir, monthly_dir)


            print('Deleting raw files')
            files_to_delete = glob.glob(f'{raw_dir}*.nc')
            num_days = monthrange(year, month)[1]
            for file in tqdm(files_to_delete):
                file_year, file_month, file_day = file.split('_')[-2].split('-')
                file_hour = file.split('_')[-1].split(':')[0]
                if int(file_day) == num_days and int(file_hour) in [22, 23]:
                    continue
                os.remove(file)


            print('Deleting summed files')
            files_to_delete = glob.glob(f'{summed_dir}*.nc')
            for file in tqdm(files_to_delete):
                os.remove(file)


            print(f'Done with {year}-{month}')
            print()
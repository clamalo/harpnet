import xarray as xr
import numpy as np
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src.get_coordinates import get_coordinates
import src.constants as constants

def xr_to_np(tile, start_month, end_month):
    def save_array(file_path, array):
        np.save(file_path, array)

    current_month = datetime(start_month[0], start_month[1], 1)
    end_month = datetime(end_month[0], end_month[1], 1)

    while current_month <= end_month:
        year = current_month.year
        month = current_month.month
        print(f'Processing {year}-{month:02d}')

        month_ds = xr.open_dataset(os.path.join(constants.raw_dir, f'{year}-{month:02d}.nc'))

        time_index = pd.DatetimeIndex(month_ds.time.values)
        filtered_times = time_index[time_index.hour.isin([3, 6, 9, 12, 15, 18, 21, 0])]
        month_ds = month_ds.sel(time=filtered_times)

        times = month_ds.time.values

        coarse_lats_pad, coarse_lons_pad, coarse_lats, coarse_lons, fine_lats, fine_lons = get_coordinates(tile)

        month_ds = month_ds.sel(lat=slice(coarse_lats_pad[0]-0.25, coarse_lats_pad[-1]+0.25), lon=slice(coarse_lons_pad[0]-0.25, coarse_lons_pad[-1]+0.25))

        coarse_ds = month_ds.interp(lat=coarse_lats_pad, lon=coarse_lons_pad)
        fine_ds = month_ds.interp(lat=fine_lats, lon=fine_lons)

        coarse_tp = coarse_ds.tp.values.astype('float32')
        fine_tp = fine_ds.tp.values.astype('float32')

        input_path = os.path.join(constants.processed_dir, str(tile), f'input_{year}_{month:02d}.npy')
        target_path = os.path.join(constants.processed_dir, str(tile), f'target_{year}_{month:02d}.npy')
        times_path = os.path.join(constants.processed_dir, str(tile), f'times_{year}_{month:02d}.npy')

        with ThreadPoolExecutor() as executor:
            executor.submit(save_array, input_path, coarse_tp)
            executor.submit(save_array, target_path, fine_tp)
            executor.submit(save_array, times_path, times)

        current_month += relativedelta(months=1)
import xarray as xr
import numpy as np
import pandas as pd
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dateutil.relativedelta import relativedelta
import zipfile
from tqdm import tqdm

from src.get_coordinates import get_coordinates
from src.constants import RAW_DIR, PROCESSED_DIR, ZIP_DIR, HOUR_INCREMENT


def xr_to_np(start_month, end_month):

    def save_array(file_path, array):
        np.save(file_path, array)

    current_month = datetime(start_month[0], start_month[1], 1)
    end_month = datetime(end_month[0], end_month[1], 1)

    while current_month <= end_month:
        year = current_month.year
        month = current_month.month
        print(f'Processing {year}-{month:02d}')

        month_ds = xr.open_dataset(os.path.join(RAW_DIR, f'{year}-{month:02d}.nc'))

        if HOUR_INCREMENT == 3:
            time_index = pd.DatetimeIndex(month_ds.time.values)
            filtered_times = time_index[time_index.hour.isin([3, 6, 9, 12, 15, 18, 21, 0])]
            month_ds = month_ds.sel(time=filtered_times)

        coarse_tp = []
        fine_tp = []
        times_arr = []

        for tile in tqdm(range(6)):

            coarse_lats_pad, coarse_lons_pad, coarse_lats, coarse_lons, fine_lats, fine_lons = get_coordinates(tile)

            tile_month_ds = month_ds.sel(lat=slice(coarse_lats_pad[0]-0.25, coarse_lats_pad[-1]+0.25), lon=slice(coarse_lons_pad[0]-0.25, coarse_lons_pad[-1]+0.25))

            times = tile_month_ds.time.values

            tile_coarse_ds = tile_month_ds.interp(lat=coarse_lats_pad, lon=coarse_lons_pad)
            tile_fine_ds = tile_month_ds.interp(lat=fine_lats, lon=fine_lons)

            tile_coarse_tp = tile_coarse_ds.tp.values.astype('float32')
            tile_fine_tp = tile_fine_ds.tp.values.astype('float32')

            coarse_tp.append(tile_coarse_tp)
            fine_tp.append(tile_fine_tp)
            times_arr.append(times)

        # concatenate along the first axis
        coarse_tp = np.concatenate(coarse_tp, axis=0)
        fine_tp = np.concatenate(fine_tp, axis=0)
        times = np.concatenate(times_arr, axis=0)

        input_path = os.path.join(PROCESSED_DIR, f'input_{year}_{month:02d}.npy')
        target_path = os.path.join(PROCESSED_DIR, f'target_{year}_{month:02d}.npy')
        times_path = os.path.join(PROCESSED_DIR, f'times_{year}_{month:02d}.npy')

        with ThreadPoolExecutor() as executor:
            executor.submit(save_array, input_path, coarse_tp)
            executor.submit(save_array, target_path, fine_tp)
            executor.submit(save_array, times_path, times)

        current_month += relativedelta(months=1)
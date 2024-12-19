import xarray as xr
import numpy as np
import pandas as pd
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dateutil.relativedelta import relativedelta
import zipfile

from src.get_coordinates import get_coordinates
from src.constants import RAW_DIR, PROCESSED_DIR, ZIP_DIR, HOUR_INCREMENT


def xr_to_np(tiles, start_month, end_month, zip_setting=False):
    """
    Process multiple tiles at once. For each tile and month, load the raw data,
    interpolate to coarse and fine resolutions, and save input, target, times, and tile arrays.
    If zip_setting is 'load' or 'save', handle zip files accordingly.
    """

    # If tiles is a single int, convert to list for consistency
    if isinstance(tiles, int):
        tiles = [tiles]

    # Handle zip load for each tile
    if zip_setting == 'load':
        for tile in tiles:
            zip_path = os.path.join(ZIP_DIR, f"{tile}.zip")
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(PROCESSED_DIR)
        return

    # Ensure directories exist for each tile
    for tile in tiles:
        os.makedirs(os.path.join(PROCESSED_DIR, str(tile)), exist_ok=True)

    current_month = datetime(start_month[0], start_month[1], 1)
    end_month_dt = datetime(end_month[0], end_month[1], 1)

    # We'll load data month by month, then process each tile within that month
    while current_month <= end_month_dt:
        year = current_month.year
        month = current_month.month
        print(f'Processing {year}-{month:02d} for all tiles...')

        # Load the monthly dataset once
        month_ds_path = os.path.join(RAW_DIR, f'{year}-{month:02d}.nc')
        if not os.path.exists(month_ds_path):
            print(f"Raw dataset not found for {year}-{month:02d}. Skipping.")
            current_month += relativedelta(months=1)
            continue

        month_ds = xr.open_dataset(month_ds_path)

        if HOUR_INCREMENT == 3:
            time_index = pd.DatetimeIndex(month_ds.time.values)
            filtered_times = time_index[time_index.hour.isin([3, 6, 9, 12, 15, 18, 21, 0])]
            month_ds = month_ds.sel(time=filtered_times)

        times = month_ds.time.values

        # Process each tile for this month
        for tile in tiles:
            # Get tile coordinates
            coarse_lats_pad, coarse_lons_pad, coarse_lats, coarse_lons, fine_lats, fine_lons = get_coordinates(tile)

            # Select and interpolate for this tile
            tile_month_ds = month_ds.sel(lat=slice(coarse_lats_pad[0]-0.25, coarse_lats_pad[-1]+0.25),
                                         lon=slice(coarse_lons_pad[0]-0.25, coarse_lons_pad[-1]+0.25))
            coarse_ds = tile_month_ds.interp(lat=coarse_lats_pad, lon=coarse_lons_pad)
            fine_ds = tile_month_ds.interp(lat=fine_lats, lon=fine_lons)

            coarse_tp = coarse_ds.tp.values.astype('float32')
            fine_tp = fine_ds.tp.values.astype('float32')

            # Create a tile array for identification
            tile_arr = np.full(shape=(len(times),), fill_value=tile, dtype=np.int32)

            # Paths
            input_path = os.path.join(PROCESSED_DIR, str(tile), f'input_{year}_{month:02d}.npy')
            target_path = os.path.join(PROCESSED_DIR, str(tile), f'target_{year}_{month:02d}.npy')
            times_path = os.path.join(PROCESSED_DIR, str(tile), f'times_{year}_{month:02d}.npy')
            tile_path = os.path.join(PROCESSED_DIR, str(tile), f'tile_{year}_{month:02d}.npy')

            # Save arrays using ThreadPoolExecutor for concurrency
            def save_array(file_path, array):
                np.save(file_path, array)

            with ThreadPoolExecutor() as executor:
                executor.submit(save_array, input_path, coarse_tp)
                executor.submit(save_array, target_path, fine_tp)
                executor.submit(save_array, times_path, times)
                executor.submit(save_array, tile_path, tile_arr)

        current_month += relativedelta(months=1)

    # Handle zip save for each tile after processing
    if zip_setting == 'save':
        for tile in tiles:
            zip_path = os.path.join(ZIP_DIR, f"{tile}.zip")
            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(os.path.join(PROCESSED_DIR, str(tile))):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, PROCESSED_DIR)
                        zipf.write(file_path, arcname)
            shutil.rmtree(os.path.join(PROCESSED_DIR, str(tile)))

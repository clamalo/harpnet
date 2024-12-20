import xarray as xr
import numpy as np
import pandas as pd
import os
import zipfile
import gc
import random
import torch
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Tuple, Union
from tqdm import tqdm

from src.tiles import tile_coordinates
from src.constants import RAW_DIR, PROCESSED_DIR, ZIP_DIR, HOUR_INCREMENT

def xr_to_np(tiles: List[int], 
             start_month: Tuple[int,int], 
             end_month: Tuple[int,int], 
             train_test_ratio: float=0.2, 
             zip_setting: Union[str,bool]=False):
    """
    Process raw NetCDF data for multiple tiles and generate combined training and testing datasets.
    The function:
    - Iterates through a time period defined by start and end months.
    - For each month and each tile, it reads coarse and fine-resolution data from NetCDF files.
    - Interpolates precipitation data onto coarse and fine grids.
    - Splits data into training and testing sets.
    - Optionally zips/unzips the processed data.

    Args:
        tiles (List[int]): List of tile indices to process.
        start_month (Tuple[int,int]): Starting year and month (e.g., (1979, 10)).
        end_month (Tuple[int,int]): Ending year and month (e.g., (1980, 9)).
        train_test_ratio (float): Fraction of data to allocate to the test set.
        zip_setting (str or bool): 
            - 'save': after processing, zip the data and remove local .npy files.
            - 'load': load existing zipped data (no processing).
            - False: just process the data without zipping.
    """

    combined_zip_path = ZIP_DIR / "combined_dataset.zip"

    # Define output file paths
    train_input_path = PROCESSED_DIR / "combined_train_input.npy"
    train_target_path = PROCESSED_DIR / "combined_train_target.npy"
    train_times_path = PROCESSED_DIR / "combined_train_times.npy"
    train_tile_ids_path = PROCESSED_DIR / "combined_train_tile_ids.npy"

    test_input_path = PROCESSED_DIR / "combined_test_input.npy"
    test_target_path = PROCESSED_DIR / "combined_test_target.npy"
    test_times_path = PROCESSED_DIR / "combined_test_times.npy"
    test_tile_ids_path = PROCESSED_DIR / "combined_test_tile_ids.npy"

    tile_elev_path = PROCESSED_DIR / "combined_tile_elev.npy"

    # If loading from zip, just extract and return
    if zip_setting == 'load':
        if combined_zip_path.exists():
            with zipfile.ZipFile(combined_zip_path, 'r') as zip_ref:
                zip_ref.extractall(PROCESSED_DIR)
            return
        else:
            raise FileNotFoundError(f"No zip file found at {combined_zip_path} to load from.")

    # Dictionary to store data before splitting (per tile)
    tile_data = {tile: {"inputs": [], "targets": [], "times": []} for tile in tiles}

    # Calculate total number of months to process for progress bar
    start_date = datetime(start_month[0], start_month[1], 1)
    end_date = datetime(end_month[0], end_month[1], 1)
    total_months = 0
    temp_month = start_date
    while temp_month <= end_date:
        total_months += 1
        temp_month += relativedelta(months=1)
    total_iterations = total_months * len(tiles)

    # Open elevation dataset once to avoid repetitive I/O
    elevation_ds = xr.open_dataset("/Users/clamalo/downloads/elevation.nc")
    if 'X' in elevation_ds.dims and 'Y' in elevation_ds.dims:
        elevation_ds = elevation_ds.rename({'X': 'lon', 'Y': 'lat'})

    current_month = start_date

    # Progress bar over total (month * tile) combinations
    pbar = tqdm(total=total_iterations, desc="Processing data")

    # Iterate month by month
    while current_month <= end_date:
        year = current_month.year
        month = current_month.month
        print(f'Processing {year}-{month:02d}')

        file_path = RAW_DIR / f'{year}-{month:02d}.nc'
        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist. Skipping.")
            current_month += relativedelta(months=1)
            continue

        # Load monthly dataset
        month_ds = xr.open_dataset(file_path)

        # If HOUR_INCREMENT=3, filter times to 3-hourly steps
        if HOUR_INCREMENT == 3:
            time_index = pd.DatetimeIndex(month_ds.time.values)
            filtered_times = time_index[time_index.hour.isin([0, 3, 6, 9, 12, 15, 18, 21])]
            month_ds = month_ds.sel(time=filtered_times)

        times = month_ds.time.values

        # Process each tile for this month
        for tile in tiles:
            coarse_latitudes, coarse_longitudes, fine_latitudes, fine_longitudes = tile_coordinates(tile)

            # Interpolate coarse and fine fields from month_ds
            coarse_ds = month_ds.interp(lat=coarse_latitudes, lon=coarse_longitudes)
            fine_ds = month_ds.interp(lat=fine_latitudes, lon=fine_longitudes)

            # Extract precipitation
            coarse_tp = coarse_ds.tp.values.astype('float32')
            fine_tp = fine_ds.tp.values.astype('float32')

            # Store raw arrays before splitting
            tile_data[tile]["inputs"].append(coarse_tp)
            tile_data[tile]["targets"].append(fine_tp)
            tile_data[tile]["times"].append(times)

            # Update progress bar after each tile is processed
            pbar.update(1)

        current_month += relativedelta(months=1)

    pbar.close()

    # Compute elevation arrays for each tile at fine resolution
    print("Computing elevation per tile...")
    num_tiles = len(tiles)
    sample_tile = tiles[0]
    _, _, fine_latitudes_sample, fine_longitudes_sample = tile_coordinates(sample_tile)
    Hf = len(fine_latitudes_sample)
    Wf = len(fine_longitudes_sample)

    tile_elev_all = np.zeros((num_tiles, 1, Hf, Wf), dtype='float32')
    for i, t in enumerate(tiles):
        _, _, fine_latitudes, fine_longitudes = tile_coordinates(t)
        elev_fine = elevation_ds.interp(lat=fine_latitudes, lon=fine_longitudes).topo.fillna(0.0).values.astype('float32')
        # Normalize elevation by highest elevation on Earth (Everest ~8848.9 m)
        elev_fine = elev_fine / 8848.9
        tile_elev_all[i, 0, :, :] = elev_fine

    # Split data into train/test sets for each tile, then combine
    print("Processing train/test splits...")
    all_train_inputs = []
    all_train_targets = []
    all_train_times = []
    all_train_tile_ids = []

    all_test_inputs = []
    all_test_targets = []
    all_test_times = []
    all_test_tile_ids = []

    # Shuffle and split each tile's dataset
    for tile_idx, tile in enumerate(tiles):
        inputs_arr = np.concatenate(tile_data[tile]["inputs"], axis=0)
        targets_arr = np.concatenate(tile_data[tile]["targets"], axis=0)
        times_arr = np.concatenate(tile_data[tile]["times"], axis=0)

        times_s = times_arr.astype('datetime64[s]').astype(np.int64)

        # Ensure 4D shape: (T, C, H, W)
        if len(inputs_arr.shape) == 3:
            inputs_arr = np.expand_dims(inputs_arr, axis=1)
        if len(targets_arr.shape) == 3:
            targets_arr = np.expand_dims(targets_arr, axis=1)

        # Shuffle samples
        indices = np.random.permutation(len(times_s))
        inputs_arr = inputs_arr[indices]
        targets_arr = targets_arr[indices]
        times_s = times_s[indices]

        # Split into train/test
        split_idx = int(train_test_ratio * len(inputs_arr))
        test_input_tile = inputs_arr[:split_idx]
        train_input_tile = inputs_arr[split_idx:]
        test_target_tile = targets_arr[:split_idx]
        train_target_tile = targets_arr[split_idx:]
        test_times_tile = times_s[:split_idx]
        train_times_tile = times_s[split_idx:]

        # Create tile ID arrays
        train_tile_ids_tile = np.full(train_input_tile.shape[0], tile, dtype=np.int32)
        test_tile_ids_tile = np.full(test_input_tile.shape[0], tile, dtype=np.int32)

        # Append to global lists
        all_train_inputs.append(train_input_tile)
        all_train_targets.append(train_target_tile)
        all_train_times.append(train_times_tile)
        all_train_tile_ids.append(train_tile_ids_tile)

        all_test_inputs.append(test_input_tile)
        all_test_targets.append(test_target_tile)
        all_test_times.append(test_times_tile)
        all_test_tile_ids.append(test_tile_ids_tile)

    # Combine all tiles into final training/testing arrays
    print("Combining all tiles...")
    train_inputs_all = np.concatenate(all_train_inputs, axis=0)
    train_targets_all = np.concatenate(all_train_targets, axis=0)
    train_times_all = np.concatenate(all_train_times, axis=0)
    train_tile_ids_all = np.concatenate(all_train_tile_ids, axis=0)

    test_inputs_all = np.concatenate(all_test_inputs, axis=0)
    test_targets_all = np.concatenate(all_test_targets, axis=0)
    test_times_all = np.concatenate(all_test_times, axis=0)
    test_tile_ids_all = np.concatenate(all_test_tile_ids, axis=0)

    # Save all arrays to .npy files
    print("Saving arrays...")
    np.save(train_input_path, train_inputs_all)
    np.save(train_target_path, train_targets_all)
    np.save(train_times_path, train_times_all)
    np.save(train_tile_ids_path, train_tile_ids_all)

    np.save(test_input_path, test_inputs_all)
    np.save(test_target_path, test_targets_all)
    np.save(test_times_path, test_times_all)
    np.save(test_tile_ids_path, test_tile_ids_all)

    np.save(tile_elev_path, tile_elev_all)

    # Cleanup memory
    del (train_inputs_all, train_targets_all, train_times_all, train_tile_ids_all,
         test_inputs_all, test_targets_all, test_times_all, test_tile_ids_all, tile_elev_all)
    gc.collect()

    # If zip_setting='save', zip the processed data and then remove the npy files
    if zip_setting == 'save':
        print("Zipping processed data...")
        with zipfile.ZipFile(combined_zip_path, 'w', compression=zipfile.ZIP_STORED) as zipf:
            zipf.write(train_input_path, "combined_train_input.npy")
            zipf.write(train_target_path, "combined_train_target.npy")
            zipf.write(train_times_path, "combined_train_times.npy")
            zipf.write(train_tile_ids_path, "combined_train_tile_ids.npy")

            zipf.write(test_input_path, "combined_test_input.npy")
            zipf.write(test_target_path, "combined_test_target.npy")
            zipf.write(test_times_path, "combined_test_times.npy")
            zipf.write(test_tile_ids_path, "combined_test_tile_ids.npy")

            zipf.write(tile_elev_path, "combined_tile_elev.npy")

        # Remove local npy files after zipping
        os.remove(train_input_path)
        os.remove(train_target_path)
        os.remove(train_times_path)
        os.remove(train_tile_ids_path)

        os.remove(test_input_path)
        os.remove(test_target_path)
        os.remove(test_times_path)
        os.remove(test_tile_ids_path)

        os.remove(tile_elev_path)
        print("Zipped and removed local npy files.")
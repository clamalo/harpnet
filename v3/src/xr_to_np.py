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
from src.constants import RAW_DIR, PROCESSED_DIR, ZIP_DIR, HOUR_INCREMENT, RANDOM_SEED

def xr_to_np(tiles: List[int], 
             start_month: Tuple[int,int], 
             end_month: Tuple[int,int], 
             train_test_ratio: float=0.2, 
             zip_setting: Union[str,bool]=False):
    """
    Process raw NetCDF data for multiple tiles and generate combined training and testing datasets incrementally.
    According to the new specifications:
    - For each month, we determine test times first, ensuring that every tile in that month uses the same test times.
    - Exactly 20% of the times are set aside for test, and the rest for train.
    - All tiles share the same test times for that month.
    - Print out requested info for tile 0 in a nice human-readable format.
    - Save monthly chunks to disk.
    - After all months are processed, concatenate into final combined arrays.
    """

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    combined_zip_path = ZIP_DIR / "combined_dataset.zip"

    # Define output file paths for final combined arrays
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

    # Ensure output directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Prepare temporary directories for monthly chunks
    monthly_dir = PROCESSED_DIR / "monthly_chunks"
    if monthly_dir.exists():
        # Clean up old monthly chunks if any
        for f in monthly_dir.iterdir():
            f.unlink()
    else:
        monthly_dir.mkdir()

    # Open elevation dataset once
    elevation_ds = xr.open_dataset("/Users/clamalo/downloads/elevation.nc")
    if 'X' in elevation_ds.dims and 'Y' in elevation_ds.dims:
        elevation_ds = elevation_ds.rename({'X': 'lon', 'Y': 'lat'})

    # Compute elevation arrays once for all tiles
    print("Computing elevation per tile...")
    sample_tile = tiles[0]
    _, _, fine_latitudes_sample, fine_longitudes_sample = tile_coordinates(sample_tile)
    Hf = len(fine_latitudes_sample)
    Wf = len(fine_longitudes_sample)
    tile_elev_all = np.zeros((len(tiles), 1, Hf, Wf), dtype='float32')
    for i, t in enumerate(tiles):
        _, _, fine_latitudes, fine_longitudes = tile_coordinates(t)
        elev_fine = elevation_ds.interp(lat=fine_latitudes, lon=fine_longitudes).topo.fillna(0.0).values.astype('float32')
        elev_fine = elev_fine / 8848.9
        tile_elev_all[i, 0, :, :] = elev_fine

    # Save elevation immediately
    np.save(tile_elev_path, tile_elev_all)

    start_date = datetime(start_month[0], start_month[1], 1)
    end_date = datetime(end_month[0], end_month[1], 1)

    # Count total months
    total_months = 0
    temp_month = start_date
    while temp_month <= end_date:
        total_months += 1
        temp_month += relativedelta(months=1)

    pbar = tqdm(total=total_months * len(tiles), desc="Processing data month-by-month")

    month_counter = 0
    monthly_train_files = []
    monthly_test_files = []

    # Process each month separately
    current_month = start_date
    while current_month <= end_date:
        year = current_month.year
        month = current_month.month
        file_path = RAW_DIR / f'{year}-{month:02d}.nc'
        if not file_path.exists():
            # No data for this month, still update progress for each tile
            for _ in tiles:
                pbar.update(1)
            current_month += relativedelta(months=1)
            continue

        month_ds = xr.open_dataset(file_path)

        # Filter times if needed
        if HOUR_INCREMENT == 3:
            time_index = pd.DatetimeIndex(month_ds.time.values)
            filtered_times = time_index[time_index.hour.isin([0, 3, 6, 9, 12, 15, 18, 21])]
            month_ds = month_ds.sel(time=filtered_times)

        times = month_ds.time.values
        T = len(times)
        if T == 0:
            # No valid times
            for _ in tiles:
                pbar.update(1)
            current_month += relativedelta(months=1)
            continue

        # Determine which times are for test:
        test_count = int(train_test_ratio * T)
        time_indices = np.arange(T)
        np.random.seed(RANDOM_SEED + month_counter)
        np.random.shuffle(time_indices)
        test_time_indices = time_indices[:test_count]
        train_time_indices = time_indices[test_count:]
        
        # Create a set of test times for quick membership check
        test_times_set = set(times[test_time_indices])
        
        # Prepare arrays to store combined train/test data for the month
        month_train_input = []
        month_train_target = []
        month_train_times = []
        month_train_tile_ids = []

        month_test_input = []
        month_test_target = []
        month_test_times = []
        month_test_tile_ids = []

        # Process each tile for this month
        for tile in tiles:
            coarse_latitudes, coarse_longitudes, fine_latitudes, fine_longitudes = tile_coordinates(tile)
            coarse_ds = month_ds.interp(lat=coarse_latitudes, lon=coarse_longitudes)
            fine_ds = month_ds.interp(lat=fine_latitudes, lon=fine_longitudes)

            coarse_tp = coarse_ds.tp.values.astype('float32')
            fine_tp = fine_ds.tp.values.astype('float32')

            # Ensure shape: (T, C, H, W)
            if len(coarse_tp.shape) == 3:
                coarse_tp = coarse_tp[:, np.newaxis, :, :]  # (T,1,Hc,Wc)
            if len(fine_tp.shape) == 3:
                fine_tp = fine_tp[:, np.newaxis, :, :]      # (T,1,Hf,Wf)

            time64 = times.astype('datetime64[s]').astype(np.int64)
            time_as_datetime = times.astype('datetime64[ns]')

            test_mask = np.array([t in test_times_set for t in time_as_datetime])
            train_mask = ~test_mask

            # Train samples for this tile
            month_train_input.append(coarse_tp[train_mask])
            month_train_target.append(fine_tp[train_mask])
            month_train_times.append(time64[train_mask])
            month_train_tile_ids.append(np.full(train_mask.sum(), tile, dtype=np.int32))

            # Test samples for this tile
            month_test_input.append(coarse_tp[test_mask])
            month_test_target.append(fine_tp[test_mask])
            month_test_times.append(time64[test_mask])
            month_test_tile_ids.append(np.full(test_mask.sum(), tile, dtype=np.int32))

            pbar.update(1)

        # Concatenate train/test arrays for this month across all tiles
        if month_train_input:
            train_input_all = np.concatenate(month_train_input, axis=0)
            train_target_all = np.concatenate(month_train_target, axis=0)
            train_times_all = np.concatenate(month_train_times, axis=0)
            train_tile_ids_all = np.concatenate(month_train_tile_ids, axis=0)

            test_input_all = np.concatenate(month_test_input, axis=0)
            test_target_all = np.concatenate(month_test_target, axis=0)
            test_times_all = np.concatenate(month_test_times, axis=0)
            test_tile_ids_all = np.concatenate(month_test_tile_ids, axis=0)

            # Save monthly chunks to temporary files
            month_train_file_prefix = monthly_dir / f"{year}_{month:02d}_train"
            month_test_file_prefix = monthly_dir / f"{year}_{month:02d}_test"

            np.save(str(month_train_file_prefix) + "_input.npy", train_input_all)
            np.save(str(month_train_file_prefix) + "_target.npy", train_target_all)
            np.save(str(month_train_file_prefix) + "_times.npy", train_times_all)
            np.save(str(month_train_file_prefix) + "_tile_ids.npy", train_tile_ids_all)

            np.save(str(month_test_file_prefix) + "_input.npy", test_input_all)
            np.save(str(month_test_file_prefix) + "_target.npy", test_target_all)
            np.save(str(month_test_file_prefix) + "_times.npy", test_times_all)
            np.save(str(month_test_file_prefix) + "_tile_ids.npy", test_tile_ids_all)

            monthly_train_files.append(str(month_train_file_prefix))
            monthly_test_files.append(str(month_test_file_prefix))

            # Cleanup arrays for this month
            del (train_input_all, train_target_all, train_times_all, train_tile_ids_all,
                 test_input_all, test_target_all, test_times_all, test_tile_ids_all,
                 month_train_input, month_train_target, month_train_times, month_train_tile_ids,
                 month_test_input, month_test_target, month_test_times, month_test_tile_ids)
            gc.collect()

        # Move to next month
        month_counter += 1
        current_month += relativedelta(months=1)

    pbar.close()

    print("Combining monthly chunks into final arrays...")

    def concatenate_npy_files(file_prefixes, suffix):
        # suffix like "_input.npy", "_target.npy", etc.

        # First pass: determine total size and verify consistency
        total_samples = 0
        shape = None
        dtype = None
        for fp in file_prefixes:
            arr = np.load(fp + suffix)
            if shape is None:
                shape = arr.shape[1:]
                dtype = arr.dtype
            total_samples += arr.shape[0]

        if total_samples == 0:
            return np.empty((0,) + shape, dtype=dtype)

        final_arr = np.empty((total_samples,) + shape, dtype=dtype)

        start = 0
        # Add a tqdm progress bar for merging files
        with tqdm(total=len(file_prefixes), desc=f"Merging {suffix} files", unit="file") as merge_pbar:
            for fp in file_prefixes:
                arr = np.load(fp + suffix)
                length = arr.shape[0]
                final_arr[start:start+length] = arr
                start += length
                merge_pbar.update(1)

        return final_arr

    # Train sets
    if monthly_train_files:
        final_train_input = concatenate_npy_files(monthly_train_files, "_input.npy")
        final_train_target = concatenate_npy_files(monthly_train_files, "_target.npy")
        final_train_times = concatenate_npy_files(monthly_train_files, "_times.npy")
        final_train_tile_ids = concatenate_npy_files(monthly_train_files, "_tile_ids.npy")
    else:
        # No training data at all
        final_train_input = np.empty((0,1,1,1), dtype=np.float32)
        final_train_target = np.empty((0,1,1,1), dtype=np.float32)
        final_train_times = np.empty((0,), dtype=np.int64)
        final_train_tile_ids = np.empty((0,), dtype=np.int32)

    # Test sets
    if monthly_test_files:
        final_test_input = concatenate_npy_files(monthly_test_files, "_input.npy")
        final_test_target = concatenate_npy_files(monthly_test_files, "_target.npy")
        final_test_times = concatenate_npy_files(monthly_test_files, "_times.npy")
        final_test_tile_ids = concatenate_npy_files(monthly_test_files, "_tile_ids.npy")
    else:
        # No test data at all
        final_test_input = np.empty((0,1,1,1), dtype=np.float32)
        final_test_target = np.empty((0,1,1,1), dtype=np.float32)
        final_test_times = np.empty((0,), dtype=np.int64)
        final_test_tile_ids = np.empty((0,), dtype=np.int32)

    # Save final arrays
    np.save(train_input_path, final_train_input)
    np.save(train_target_path, final_train_target)
    np.save(train_times_path, final_train_times)
    np.save(train_tile_ids_path, final_train_tile_ids)

    np.save(test_input_path, final_test_input)
    np.save(test_target_path, final_test_target)
    np.save(test_times_path, final_test_times)
    np.save(test_tile_ids_path, final_test_tile_ids)

    # Cleanup large arrays
    del (final_train_input, final_train_target, final_train_times, final_train_tile_ids,
         final_test_input, final_test_target, final_test_times, final_test_tile_ids)
    gc.collect()

    # Optionally zip if zip_setting=='save'
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
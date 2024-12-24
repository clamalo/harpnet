"""
Converts raw NetCDF data into NumPy arrays for training/testing. 
Performs a temporal split (train vs. test) and computes mean/std normalization stats.
Also supports optional compression into a single .npz file based on ZIP_SETTING.
"""

import xarray as xr
import numpy as np
import pandas as pd
import os
import gc
import random
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional
from tqdm import tqdm

from src.tiles import tile_coordinates
from src.constants import (
    RAW_DIR,
    PROCESSED_DIR,
    HOUR_INCREMENT,
    RANDOM_SEED,
    DATA_START_MONTH,
    DATA_END_MONTH,
    NORMALIZATION_STATS_FILE,
    TILES,
    ZIP_SETTING,
    SAVE_PRECISION
)

def xr_to_np():
    """
    Converts NetCDF files (RAW_DIR) to memory-mapped NumPy arrays (PROCESSED_DIR).
    Splits data temporally into train/test sets (80%/20%), then calculates mean/std from the training target.
    If ZIP_SETTING=='save', compresses everything into a single .npz and removes local .npy files.
    If ZIP_SETTING=='load', loads from .npz into .npy and exits.
    """

    # --- SEED & PRECISION VALIDATION ---
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    ALLOWED_PRECISIONS = ("float16", "float32")
    if SAVE_PRECISION not in ALLOWED_PRECISIONS:
        raise ValueError(f"SAVE_PRECISION must be one of {ALLOWED_PRECISIONS}, but got '{SAVE_PRECISION}'.")

    # --- PREP OUTPUT PATHS ---
    train_input_path = PROCESSED_DIR / "combined_train_input.npy"
    train_target_path = PROCESSED_DIR / "combined_train_target.npy"
    train_times_path = PROCESSED_DIR / "combined_train_times.npy"
    train_tile_ids_path = PROCESSED_DIR / "combined_train_tile_ids.npy"
    test_input_path = PROCESSED_DIR / "combined_test_input.npy"
    test_target_path = PROCESSED_DIR / "combined_test_target.npy"
    test_times_path = PROCESSED_DIR / "combined_test_times.npy"
    test_tile_ids_path = PROCESSED_DIR / "combined_test_tile_ids.npy"
    tile_elev_path = PROCESSED_DIR / "combined_tile_elev.npy"
    normalization_stats_path = NORMALIZATION_STATS_FILE

    # --- LOAD (IF ZIP_SETTING == 'load') ---
    if ZIP_SETTING == 'load':
        npz_path = PROCESSED_DIR / "combined_dataset.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"No compressed NPZ file found at {npz_path} to load from.")

        logging.info(f"Loading data from {npz_path}...")
        with np.load(npz_path, allow_pickle=True) as data:
            np.save(train_input_path, data["train_input"])
            np.save(train_target_path, data["train_target"])
            np.save(train_times_path, data["train_times"])
            np.save(train_tile_ids_path, data["train_tile_ids"])
            np.save(test_input_path, data["test_input"])
            np.save(test_target_path, data["test_target"])
            np.save(test_times_path, data["test_times"])
            np.save(test_tile_ids_path, data["test_tile_ids"])
            np.save(tile_elev_path, data["tile_elev"])
            np.save(normalization_stats_path, data["normalization_stats"])
        logging.info("NPZ data successfully loaded into .npy files. Exiting.")
        return

    # --- ENSURE PROCESSED_DIR EXISTS ---
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # --- DATE RANGE HANDLING ---
    start_date = datetime(DATA_START_MONTH[0], DATA_START_MONTH[1], 1)
    end_date = datetime(DATA_END_MONTH[0], DATA_END_MONTH[1], 1)
    total_months = 0
    temp_month = start_date
    while temp_month <= end_date:
        total_months += 1
        temp_month += relativedelta(months=1)
    train_end_idx = int(0.8 * total_months)

    # For logging: Gather a full list of months, then split into train vs. test
    month_list = []
    temp_month_for_list = start_date
    while temp_month_for_list <= end_date:
        month_list.append(temp_month_for_list)
        temp_month_for_list += relativedelta(months=1)

    train_months = month_list[:train_end_idx]
    test_months = month_list[train_end_idx:]

    # Print the first/last month and the count in each set
    if len(train_months) > 0:
        tm_first = train_months[0]
        tm_last = train_months[-1]
        logging.info(
            f"Train months: {tm_first.year}-{tm_first.month:02d} to {tm_last.year}-{tm_last.month:02d} "
            f"({len(train_months)} months)"
        )
    else:
        logging.info("Train months: None (0 months)")

    if len(test_months) > 0:
        tm_first = test_months[0]
        tm_last = test_months[-1]
        logging.info(
            f"Test months: {tm_first.year}-{tm_first.month:02d} to {tm_last.year}-{tm_last.month:02d} "
            f"({len(test_months)} months)"
        )
    else:
        logging.info("Test months: None (0 months)")

    # --- FIRST PASS: COUNT TRAIN/TEST SAMPLES ---
    train_count = 0
    test_count = 0
    month_idx = 0
    current_month = start_date
    while current_month <= end_date:
        year, month = current_month.year, current_month.month
        file_path = RAW_DIR / f'{year}-{month:02d}.nc'
        if file_path.exists():
            with xr.open_dataset(file_path) as ds:
                if HOUR_INCREMENT == 3:
                    time_index = pd.DatetimeIndex(ds.time.values)
                    selected_hours = time_index[time_index.hour.isin([0, 3, 6, 9, 12, 15, 18, 21])]
                    ds = ds.sel(time=selected_hours)
                T = len(ds.time)
                samples_this_month = T * len(TILES)
            if month_idx < train_end_idx:
                train_count += samples_this_month
            else:
                test_count += samples_this_month

        # Fix: use relativedelta here, not relativedata
        current_month += relativedelta(months=1)
        month_idx += 1

    logging.info(f"Train samples: {train_count}  |  Test samples: {test_count}")
    if train_count == 0 and test_count == 0:
        logging.warning("No data found in any months. Exiting xr_to_np early.")
        return

    # --- ALLOCATE MEMMAP ARRAYS (INITIAL, MAY RESHAPE) ---
    train_input_map = np.memmap(train_input_path, mode='w+', dtype=SAVE_PRECISION, shape=(train_count, 1, 1, 1))
    train_target_map = np.memmap(train_target_path, mode='w+', dtype=SAVE_PRECISION, shape=(train_count, 1, 1, 1))
    train_times_map = np.memmap(train_times_path, mode='w+', dtype='int64', shape=(train_count,))
    train_tileids_map = np.memmap(train_tile_ids_path, mode='w+', dtype='int32', shape=(train_count,))
    test_input_map = np.memmap(test_input_path, mode='w+', dtype=SAVE_PRECISION, shape=(test_count, 1, 1, 1))
    test_target_map = np.memmap(test_target_path, mode='w+', dtype=SAVE_PRECISION, shape=(test_count, 1, 1, 1))
    test_times_map = np.memmap(test_times_path, mode='w+', dtype='int64', shape=(test_count,))
    test_tileids_map = np.memmap(test_tile_ids_path, mode='w+', dtype='int32', shape=(test_count,))

    # --- DETERMINE SHAPES FOR COARSE & FINE TILES ---
    def get_coarse_shape_for_one_tile(ds: xr.Dataset, tile: int):
        clat, clon, _, _ = tile_coordinates(tile)
        c_ds = ds.interp(lat=clat, lon=clon)
        c_tp = c_ds.tp.values
        return c_tp.shape[1], c_tp.shape[2]

    def get_fine_shape_for_one_tile(ds: xr.Dataset, tile: int):
        _, _, flat, flon = tile_coordinates(tile)
        f_ds = ds.interp(lat=flat, lon=flon)
        f_tp = f_ds.tp.values
        return f_tp.shape[1], f_tp.shape[2]

    found_shape = False
    cH, cW = 0, 0
    fH, fW = 0, 0
    m_idx = 0
    cmonth = start_date
    while cmonth <= end_date and not found_shape:
        fpath = RAW_DIR / f"{cmonth.year}-{cmonth.month:02d}.nc"
        if fpath.exists():
            with xr.open_dataset(fpath) as ds_shape:
                if HOUR_INCREMENT == 3:
                    tdx = pd.DatetimeIndex(ds_shape.time.values)
                    selected_hrs = tdx[tdx.hour.isin([0,3,6,9,12,15,18,21])]
                    ds_shape = ds_shape.sel(time=selected_hrs)
                if len(ds_shape.time) > 0:
                    for tile_ in TILES:
                        cH, cW = get_coarse_shape_for_one_tile(ds_shape, tile_)
                        fH, fW = get_fine_shape_for_one_tile(ds_shape, tile_)
                        found_shape = True
                        break
        cmonth += relativedelta(months=1)
        m_idx += 1

    if not found_shape:
        logging.warning("No data found for any valid tile. Exiting.")
        return

    # --- RE-CREATE MEMMAPS WITH ACTUAL SHAPES ---
    train_input_map = np.memmap(train_input_path, mode='w+', dtype=SAVE_PRECISION, shape=(train_count, 1, cH, cW))
    train_target_map = np.memmap(train_target_path, mode='w+', dtype=SAVE_PRECISION, shape=(train_count, 1, fH, fW))
    test_input_map  = np.memmap(test_input_path,  mode='w+', dtype=SAVE_PRECISION, shape=(test_count, 1, cH, cW))
    test_target_map = np.memmap(test_target_path, mode='w+', dtype=SAVE_PRECISION, shape=(test_count, 1, fH, fW))

    train_times_map = np.memmap(train_times_path, mode='w+', dtype='int64', shape=(train_count,))
    train_tileids_map = np.memmap(train_tile_ids_path, mode='w+', dtype='int32', shape=(train_count,))
    test_times_map  = np.memmap(test_times_path,  mode='w+', dtype='int64', shape=(test_count,))
    test_tileids_map= np.memmap(test_tile_ids_path, mode='w+', dtype='int32', shape=(test_count,))

    # --- SECOND PASS: FILL ARRAYS ---
    train_write_index = 0
    test_write_index = 0
    month_idx = 0
    current_month = start_date
    pbar = tqdm(total=total_months * len(TILES), desc="Processing data (2nd pass)")

    while current_month <= end_date:
        year, month = current_month.year, current_month.month
        file_path = RAW_DIR / f'{year}-{month:02d}.nc'
        if not file_path.exists():
            for _ in TILES:
                pbar.update(1)
            current_month += relativedelta(months=1)
            month_idx += 1
            continue

        with xr.open_dataset(file_path) as ds_month:
            if HOUR_INCREMENT == 3:
                tindex = pd.DatetimeIndex(ds_month.time.values)
                sel_hours = tindex[tindex.hour.isin([0,3,6,9,12,15,18,21])]
                ds_month = ds_month.sel(time=sel_hours)
            times = ds_month.time.values
            T = len(times)
            if T == 0:
                for _ in TILES:
                    pbar.update(1)
                current_month += relativedelta(months=1)
                month_idx += 1
                continue

            times_64 = times.astype("datetime64[s]").astype(np.int64)
            is_train_month = (month_idx < train_end_idx)

            for tile_ in TILES:
                clat, clon, flat, flon = tile_coordinates(tile_)
                c_ds = ds_month.interp(lat=clat, lon=clon)
                c_tp = c_ds.tp.values.astype(SAVE_PRECISION)
                c_tp = c_tp[:, np.newaxis, :, :]

                f_ds = ds_month.interp(lat=flat, lon=flon)
                f_tp = f_ds.tp.values.astype(SAVE_PRECISION)
                f_tp = f_tp[:, np.newaxis, :, :]

                tile_ids_arr = np.full(shape=(T,), fill_value=tile_, dtype=np.int32)

                if is_train_month:
                    end_ = train_write_index + T
                    train_input_map[train_write_index:end_, :, :, :] = c_tp
                    train_target_map[train_write_index:end_, :, :, :] = f_tp
                    train_times_map[train_write_index:end_:] = times_64
                    train_tileids_map[train_write_index:end_:] = tile_ids_arr
                    train_write_index = end_
                else:
                    end_ = test_write_index + T
                    test_input_map[test_write_index:end_, :, :, :] = c_tp
                    test_target_map[test_write_index:end_, :, :, :] = f_tp
                    test_times_map[test_write_index:end_:] = times_64
                    test_tileids_map[test_write_index:end_:] = tile_ids_arr
                    test_write_index = end_

                pbar.update(1)

        current_month += relativedelta(months=1)
        month_idx += 1

    pbar.close()

    # --- COMPUTE NORMALIZATION STATS FROM TRAINING TARGET ---
    train_input_map.flush()
    train_target_map.flush()
    train_times_map.flush()
    train_tileids_map.flush()
    test_input_map.flush()
    test_target_map.flush()
    test_times_map.flush()
    test_tileids_map.flush()

    if train_write_index > 0:
        logging.info("Computing mean/std from training target in streaming mode...")
        train_target_map_ro = np.memmap(train_target_path, mode='r', dtype=SAVE_PRECISION,
                                        shape=(train_write_index, 1, fH, fW))

        sum_ = 0.0
        sum_sq_ = 0.0
        total_pixels = 0
        batch_size_for_stats = 50000

        for start_idx in range(0, train_write_index, batch_size_for_stats):
            end_idx = min(start_idx + batch_size_for_stats, train_write_index)
            data_slice = train_target_map_ro[start_idx:end_idx].astype(np.float64, copy=False).reshape(-1)
            sum_ += float(data_slice.sum())
            sum_sq_ += float((data_slice * data_slice).sum())
            total_pixels += data_slice.size

        if total_pixels < 2:
            mean_val = sum_ / max(total_pixels, 1)
            std_val = 1e-8
        else:
            mean_val = sum_ / total_pixels
            var_ = (sum_sq_ / total_pixels) - (mean_val ** 2)
            std_val = float(np.sqrt(max(var_, 1e-8)))
    else:
        logging.warning("No training samples found => Using mean=0, std=1")
        mean_val = 0.0
        std_val = 1.0

    # --- SAVE NORMALIZATION STATS ---
    np.save(normalization_stats_path, np.array([mean_val, std_val], dtype=SAVE_PRECISION))
    logging.info(f"Normalization stats saved (dtype={SAVE_PRECISION}): mean={mean_val:.5f}, std={std_val:.5f}")

    # --- OPEN ELEVATION & PREPARE TILE-ELEV ARRAY ---
    logging.info("Computing elevation per tile...")
    elevation_ds = xr.open_dataset("/Users/clamalo/downloads/elevation.nc")
    if 'X' in elevation_ds.dims and 'Y' in elevation_ds.dims:
        elevation_ds = elevation_ds.rename({'X': 'lon', 'Y': 'lat'})

    sample_tile = TILES[0]
    _, _, fine_lat_sample, fine_lon_sample = tile_coordinates(sample_tile)
    Hf = len(fine_lat_sample)
    Wf = len(fine_lon_sample)
    tile_elev_all = np.zeros((len(TILES), 1, Hf, Wf), dtype=SAVE_PRECISION)
    for i, t in enumerate(TILES):
        _, _, fine_lats, fine_lons = tile_coordinates(t)
        elev_fine = elevation_ds.interp(lat=fine_lats, lon=fine_lons).topo.fillna(0.0).values
        elev_fine = (elev_fine / 8848.9).astype(SAVE_PRECISION)
        tile_elev_all[i, 0, :, :] = elev_fine
    elevation_ds.close()
    np.save(tile_elev_path, tile_elev_all)

    # --- OPTIONAL .NPZ COMPRESSION ---
    if ZIP_SETTING == 'save':
        npz_path = PROCESSED_DIR / "combined_dataset.npz"
        logging.info(f"Saving to compressed NPZ: {npz_path}")

        tr_in = np.memmap(train_input_path, mode='r', dtype=SAVE_PRECISION, shape=(train_write_index, 1, cH, cW))
        tr_tg = np.memmap(train_target_path, mode='r', dtype=SAVE_PRECISION, shape=(train_write_index, 1, fH, fW))
        tr_tm = np.memmap(train_times_path, mode='r', dtype='int64', shape=(train_write_index,))
        tr_tid= np.memmap(train_tile_ids_path, mode='r', dtype='int32', shape=(train_write_index,))

        te_in = np.memmap(test_input_path, mode='r', dtype=SAVE_PRECISION, shape=(test_write_index, 1, cH, cW))
        te_tg = np.memmap(test_target_path, mode='r', dtype=SAVE_PRECISION, shape=(test_write_index, 1, fH, fW))
        te_tm = np.memmap(test_times_path, mode='r', dtype='int64', shape=(test_write_index,))
        te_tid= np.memmap(test_tile_ids_path, mode='r', dtype='int32', shape=(test_write_index,))

        tile_elev = np.load(tile_elev_path)
        norm_stats = np.array([mean_val, std_val], dtype=SAVE_PRECISION)

        np.savez_compressed(
            npz_path,
            train_input=tr_in,
            train_target=tr_tg,
            train_times=tr_tm,
            train_tile_ids=tr_tid,
            test_input=te_in,
            test_target=te_tg,
            test_times=te_tm,
            test_tile_ids=te_tid,
            tile_elev=tile_elev,
            normalization_stats=norm_stats
        )

        logging.info("Data compressed and saved as NPZ. Removing local .npy files...")
        for f in [
            train_input_path, train_target_path, train_times_path, train_tile_ids_path,
            test_input_path, test_target_path, test_times_path, test_tile_ids_path,
            tile_elev_path, normalization_stats_path
        ]:
            if f.exists():
                os.remove(f)

        logging.info("Local .npy files removed.")

    gc.collect()
    logging.info("xr_to_np completed successfully.")
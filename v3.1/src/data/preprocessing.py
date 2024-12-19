"""
Preprocessing functions including xr_to_np conversion.
"""

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

from src.utils.get_coordinates import get_coordinates
from src.config import Config
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def xr_to_np(
    tiles: List[int],
    start_month: Tuple[int,int],
    end_month: Tuple[int,int],
    train_test_ratio: float=0.2,
    zip_setting: Union[str,bool]=False
):
    """
    Process multiple tiles and generate combined training/testing datasets.

    Args:
        tiles: List of tile indices.
        start_month: (year, month) start.
        end_month: (year, month) end.
        train_test_ratio: ratio for splitting train/test.
        zip_setting: 'save', 'load', or False for zip operations.
    """

    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)

    combined_zip_path = Config.ZIP_DIR / "combined_dataset.zip"

    train_input_path = Config.PROCESSED_DIR / "combined_train_input.npy"
    train_target_path = Config.PROCESSED_DIR / "combined_train_target.npy"
    train_times_path = Config.PROCESSED_DIR / "combined_train_times.npy"
    train_tile_ids_path = Config.PROCESSED_DIR / "combined_train_tile_ids.npy"

    test_input_path = Config.PROCESSED_DIR / "combined_test_input.npy"
    test_target_path = Config.PROCESSED_DIR / "combined_test_target.npy"
    test_times_path = Config.PROCESSED_DIR / "combined_test_times.npy"
    test_tile_ids_path = Config.PROCESSED_DIR / "combined_test_tile_ids.npy"

    tile_elev_path = Config.PROCESSED_DIR / "combined_tile_elev.npy"

    if zip_setting == 'load':
        if combined_zip_path.exists():
            logger.info("Loading preprocessed data from zip.")
            with zipfile.ZipFile(combined_zip_path, 'r') as zip_ref:
                zip_ref.extractall(Config.PROCESSED_DIR)
            return
        else:
            raise FileNotFoundError(f"No zip file found at {combined_zip_path} to load from.")

    tile_data = {tile: {"inputs": [], "targets": [], "times": []} for tile in tiles}

    current_month = datetime(start_month[0], start_month[1], 1)
    end_month_dt = datetime(end_month[0], end_month[1], 1)

    elevation_path = "/Users/clamalo/downloads/elevation.nc"
    if not os.path.exists(elevation_path):
        raise FileNotFoundError(f"Elevation file not found at {elevation_path}.")

    elevation_ds = xr.open_dataset(elevation_path)
    if 'X' in elevation_ds.dims and 'Y' in elevation_ds.dims:
        elevation_ds = elevation_ds.rename({'X': 'lon', 'Y': 'lat'})

    while current_month <= end_month_dt:
        year = current_month.year
        month = current_month.month
        logger.info(f'Processing {year}-{month:02d}')

        file_path = Config.RAW_DIR / f'{year}-{month:02d}.nc'
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist. Skipping this month.")
            current_month += relativedelta(months=1)
            continue

        month_ds = xr.open_dataset(file_path)

        if Config.HOUR_INCREMENT == 3:
            time_index = pd.DatetimeIndex(month_ds.time.values)
            filtered_times = time_index[time_index.hour.isin([0, 3, 6, 9, 12, 15, 18, 21])]
            month_ds = month_ds.sel(time=filtered_times)

        times = month_ds.time.values

        for tile in tiles:
            (coarse_lats_pad, coarse_lons_pad, 
             coarse_lats, coarse_lons, 
             fine_lats, fine_lons) = get_coordinates(tile)

            tile_ds = month_ds.sel(
                lat=slice(coarse_lats_pad[0]-0.25, coarse_lats_pad[-1]+0.25),
                lon=slice(coarse_lons_pad[0]-0.25, coarse_lons_pad[-1]+0.25)
            )

            coarse_ds = tile_ds.interp(lat=coarse_lats_pad, lon=coarse_lons_pad)
            fine_ds = tile_ds.interp(lat=fine_lats, lon=fine_lons)

            coarse_tp = coarse_ds.tp.values.astype('float32')
            fine_tp = fine_ds.tp.values.astype('float32')

            tile_data[tile]["inputs"].append(coarse_tp)
            tile_data[tile]["targets"].append(fine_tp)
            tile_data[tile]["times"].append(times)

        current_month += relativedelta(months=1)

    logger.info("Computing elevation per tile...")
    num_tiles = len(tiles)
    sample_tile = tiles[0]
    _, _, _, _, fine_lats_sample, fine_lons_sample = get_coordinates(sample_tile)
    Hf = len(fine_lats_sample)
    Wf = len(fine_lons_sample)

    tile_elev_all = np.zeros((num_tiles, 1, Hf, Wf), dtype='float32')
    for i, tile in enumerate(tiles):
        _, _, _, _, fine_lats, fine_lons = get_coordinates(tile)
        elev_fine = elevation_ds.interp(lat=fine_lats, lon=fine_lons).topo.fillna(0.0).values.astype('float32')
        elev_fine = elev_fine / 8848.9  # Normalize elevation
        tile_elev_all[i, 0, :, :] = elev_fine

    # Combine and split data
    logger.info("Processing train/test splits...")
    all_train_inputs = []
    all_train_targets = []
    all_train_times = []
    all_train_tile_ids = []

    all_test_inputs = []
    all_test_targets = []
    all_test_times = []
    all_test_tile_ids = []

    for tile_idx, tile in enumerate(tiles):
        inputs_arr = np.concatenate(tile_data[tile]["inputs"], axis=0)
        targets_arr = np.concatenate(tile_data[tile]["targets"], axis=0)
        times_arr = np.concatenate(tile_data[tile]["times"], axis=0)

        times_s = times_arr.astype('datetime64[s]').astype(np.int64)

        if len(inputs_arr.shape) == 3:
            inputs_arr = np.expand_dims(inputs_arr, axis=1)
        if len(targets_arr.shape) == 3:
            targets_arr = np.expand_dims(targets_arr, axis=1)

        indices = np.random.permutation(len(times_s))
        inputs_arr = inputs_arr[indices]
        targets_arr = targets_arr[indices]
        times_s = times_s[indices]

        split_idx = int(train_test_ratio * len(inputs_arr))
        test_input_tile = inputs_arr[:split_idx]
        train_input_tile = inputs_arr[split_idx:]
        test_target_tile = targets_arr[:split_idx]
        train_target_tile = targets_arr[split_idx:]
        test_times_tile = times_s[:split_idx]
        train_times_tile = times_s[split_idx:]

        train_tile_ids_tile = np.full(train_input_tile.shape[0], tile, dtype=np.int32)
        test_tile_ids_tile = np.full(test_input_tile.shape[0], tile, dtype=np.int32)

        all_train_inputs.append(train_input_tile)
        all_train_targets.append(train_target_tile)
        all_train_times.append(train_times_tile)
        all_train_tile_ids.append(train_tile_ids_tile)

        all_test_inputs.append(test_input_tile)
        all_test_targets.append(test_target_tile)
        all_test_times.append(test_times_tile)
        all_test_tile_ids.append(test_tile_ids_tile)

    logger.info("Combining all tiles...")
    train_inputs_all = np.concatenate(all_train_inputs, axis=0)
    train_targets_all = np.concatenate(all_train_targets, axis=0)
    train_times_all = np.concatenate(all_train_times, axis=0)
    train_tile_ids_all = np.concatenate(all_train_tile_ids, axis=0)

    test_inputs_all = np.concatenate(all_test_inputs, axis=0)
    test_targets_all = np.concatenate(all_test_targets, axis=0)
    test_times_all = np.concatenate(all_test_times, axis=0)
    test_tile_ids_all = np.concatenate(all_test_tile_ids, axis=0)

    logger.info("Saving arrays...")
    np.save(train_input_path, train_inputs_all)
    np.save(train_target_path, train_targets_all)
    np.save(train_times_path, train_times_all)
    np.save(train_tile_ids_path, train_tile_ids_all)

    np.save(test_input_path, test_inputs_all)
    np.save(test_target_path, test_targets_all)
    np.save(test_times_path, test_times_all)
    np.save(test_tile_ids_path, test_tile_ids_all)

    np.save(tile_elev_path, tile_elev_all)

    del (train_inputs_all, train_targets_all, train_times_all, train_tile_ids_all,
         test_inputs_all, test_targets_all, test_times_all, test_tile_ids_all, tile_elev_all)
    gc.collect()

    if zip_setting == 'save':
        logger.info("Zipping processed data...")
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

        os.remove(train_input_path)
        os.remove(train_target_path)
        os.remove(train_times_path)
        os.remove(train_tile_ids_path)
        os.remove(test_input_path)
        os.remove(test_target_path)
        os.remove(test_times_path)
        os.remove(test_tile_ids_path)
        os.remove(tile_elev_path)
        logger.info("Zipped and removed local npy files.")
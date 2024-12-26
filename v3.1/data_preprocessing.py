import numpy as np
import xarray as xr
from tqdm import tqdm
import os

from config import (
    RAW_DIR, PROCESSED_DIR,
    START_MONTH, END_MONTH, TRAIN_SPLIT,
    SECONDARY_TILES, TRAINING_TILES,
    SAVE_PRECISION,
    INCLUDE_ZEROS,
    ELEVATION_FILE
)
from tiles import get_tile_dict, tile_coordinates


def _month_to_int(year: int, month: int) -> int:
    """
    Convert a (year, month) tuple into a single integer index.
    For example, (1979, 1) -> 1979*12 + 1 = 23749.
    
    Args:
        year (int): The year component.
        month (int): The month component (1-12).
    
    Returns:
        int: An integer that increases with later year-month pairs.
    """
    return year * 12 + month


def _iterate_months(start=(1979, 1), end=(2020, 12)):
    """
    Generate (year, month) pairs from 'start' up to and including 'end'.
    This function yields one pair per month in ascending order.
    
    Args:
        start (tuple): A (year, month) tuple for the start date.
        end (tuple): A (year, month) tuple for the end date.

    Yields:
        (int, int): (year, month) pairs for each month in the range.
    """
    start_int = _month_to_int(*start)
    end_int = _month_to_int(*end)
    y, m = start
    while _month_to_int(y, m) <= end_int:
        yield (y, m)
        m += 1
        if m > 12:
            m = 1
            y += 1


def _split_data_by_date(
    date_array: np.ndarray,
    data_arrays: list,
    train_split
) -> dict:
    """
    Split data into training and testing sets using either:
      1. A float in [0,1] representing the fraction of data to train on.
      2. A (year, month) tuple to define a split date.

    When a float is used (e.g. 0.8), the first 80% of the samples
    are assigned to the training set, and the remaining 20% to testing.

    When a date tuple is used (e.g. (1985, 6)), all samples whose
    (year, month) occur before that date go to the training set,
    and that date and afterward go to testing.

    Note on time shape:
    - date_array should have shape (N, 4): (year, month, day, hour).
    - For date-based splits, only (year, month) are used here.

    Args:
        date_array (np.ndarray): 
            Shape (N, 4), containing [year, month, day, hour] for each sample.
        data_arrays (list of np.ndarray):
            Each array must have the same first dimension N.
            Typically [input_arr, target_arr, time_arr, tile_arr].
        train_split (float or (int, int)):
            Either a fraction in [0, 1] or a (year, month) tuple.

    Returns:
        dict: 
            {
                'train': [input_train, target_train, time_train, tile_train],
                'test':  [input_test,  target_test,  time_test,  tile_test]
            }
    """
    n_samples = date_array.shape[0]

    if isinstance(train_split, float):
        # Fraction-based splitting
        split_idx = int(train_split * n_samples)
        train_mask = np.zeros(n_samples, dtype=bool)
        train_mask[:split_idx] = True
    else:
        # Date-based splitting
        split_year, split_month = train_split
        split_int = _month_to_int(split_year, split_month)

        # Compare only year/month
        months_int = np.array([
            _month_to_int(date_array[i, 0], date_array[i, 1])
            for i in range(n_samples)
        ])
        train_mask = (months_int < split_int)

    out_dict = {'train': [], 'test': []}
    for arr in data_arrays:
        train_data = arr[train_mask]
        test_data = arr[~train_mask]
        out_dict['train'].append(train_data)
        out_dict['test'].append(test_data)

    return out_dict


def preprocess_data() -> None:
    """
    Preprocess raw monthly NetCDF files into training and testing datasets,
    using a two-pass memory-mapped approach suitable for large-scale data,
    including hourly resolution. The final output is a single .npz file 
    containing both the train and test sets, plus the tile-based elevation data.

    Workflow Steps:
    
    1. Determine which tiles to process from TRAINING_TILES. These tiles
       are defined in config.py and may represent coarse-to-fine 
       geographical domains.
    
    2. Perform a first pass to count the total number of samples 
       (hours × number of tiles × number of months) so that we can 
       allocate sufficiently large memory-mapped arrays.
       - If INCLUDE_ZEROS is True, all hours are counted.
       - If INCLUDE_ZEROS is False, we still allocate space for 
         all hours but may ultimately skip zero-value hours in the second pass.
    
    3. Allocate memory-mapped arrays for:
       - input_mm   (float32) shape: (total_samples, cLat, cLon)
       - target_mm  (float32) shape: (total_samples, fLat, fLon)
       - time_mm    (int32)   shape: (total_samples, 4)  # [year, month, day, hour]
       - tile_mm    (int32)   shape: (total_samples,)
       
       Here cLat, cLon = coarse lat/lon size, and fLat, fLon = fine lat/lon size.
    
    4. Second pass over the months/tiles:
       - For each tile in TRAINING_TILES, open the NetCDF file for that month 
         (if it exists).
       - Interpolate the 'tp' variable onto the coarse grid and the fine grid.
       - Iterate over each time index (hourly step). If INCLUDE_ZEROS is False
         and the entire coarse grid is zero at that time, skip it. Otherwise, 
         write the data and associated time/tile info into the memory map.
       - Keep track of how many total samples were actually written (idx).
    
    5. After the second pass completes, convert only the slice [0 : idx] 
       of each memory-mapped array into an in-memory array. This effectively 
       discards any unused space in the memmap if we skipped zero-value hours.
    
    6. Use _split_data_by_date to split the final arrays into train/test 
       according to TRAIN_SPLIT. This may be a fractional split or a date-based split.
    
    7. Convert the input/target arrays to the floating-point precision specified 
       by SAVE_PRECISION (either float16 or float32).
    
    8. Load elevation data from ELEVATION_FILE once. For each tile, create
       a fine-resolution elevation grid (same size as the target grid),
       and store it in a single array, tile_elevations, with shape 
       (num_tiles, fLat, fLon). Also store tile_ids in a 1D array.
    
    9. Save both train and test splits, plus tile_elevations and tile_ids, 
       in a single .npz file named combined_data.npz in PROCESSED_DIR.
    
    10. Remove the temporary .dat memmap files to clean up disk usage.
    """

    # 1) Gather tile IDs to process
    tile_dict = get_tile_dict()
    tile_ids = sorted(TRAINING_TILES)
    if not tile_ids:
        raise ValueError("No tiles provided in TRAINING_TILES; nothing to process.")

    # Determine coarse/fine grid shapes from the first tile
    def _get_tile_shapes():
        test_tid = tile_ids[0]
        lat_coarse, lon_coarse, lat_fine, lon_fine = tile_coordinates(test_tid)
        return (len(lat_coarse), len(lon_coarse)), (len(lat_fine), len(lon_fine))

    c_shape, f_shape = _get_tile_shapes()
    cLat, cLon = c_shape
    fLat, fLon = f_shape

    # Gather all months in range
    all_months = list(_iterate_months(start=START_MONTH, end=END_MONTH))

    # 2) First pass: worst-case count of total samples (assuming no skipping)
    def _count_total_samples() -> int:
        total_count = 0
        pbar = tqdm(total=len(all_months) * len(tile_ids), desc="Counting total samples")
        for (year, month) in all_months:
            nc_file = RAW_DIR / f"{year:04d}-{month:02d}.nc"
            if not nc_file.exists():
                pbar.update(len(tile_ids))
                continue

            with xr.open_dataset(nc_file) as ds:
                if ("time" not in ds.coords) or ("tp" not in ds.variables):
                    pbar.update(len(tile_ids))
                    continue

                n_time = ds.sizes.get("time", 0)
                for _ in tile_ids:
                    total_count += n_time

                pbar.update(len(tile_ids))
        pbar.close()
        return total_count

    n_total_samples = _count_total_samples()
    if n_total_samples == 0:
        raise ValueError(
            "No samples found for the specified tiles/months. "
            "Please check data files or tile configuration."
        )

    # 3) Allocate memmap arrays
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    input_mm_path = PROCESSED_DIR / "input_mm.dat"
    target_mm_path = PROCESSED_DIR / "target_mm.dat"
    time_mm_path = PROCESSED_DIR / "time_mm.dat"
    tile_mm_path = PROCESSED_DIR / "tile_mm.dat"

    for p in [input_mm_path, target_mm_path, time_mm_path, tile_mm_path]:
        if p.exists():
            os.remove(p)

    input_mm = np.memmap(
        input_mm_path, dtype=np.float32, mode='w+',
        shape=(n_total_samples, cLat, cLon)
    )
    target_mm = np.memmap(
        target_mm_path, dtype=np.float32, mode='w+',
        shape=(n_total_samples, fLat, fLon)
    )
    time_mm = np.memmap(
        time_mm_path, dtype=np.int32, mode='w+',
        shape=(n_total_samples, 4)  # year, month, day, hour
    )
    tile_mm = np.memmap(
        tile_mm_path, dtype=np.int32, mode='w+',
        shape=(n_total_samples,)
    )

    # 4) Second pass: interpolate + write, optionally skipping zeros
    idx = 0
    total_iters = len(all_months) * len(tile_ids)
    pbar = tqdm(total=total_iters, desc="Writing to memmap")

    for (year, month) in all_months:
        nc_file = RAW_DIR / f"{year:04d}-{month:02d}.nc"
        if not nc_file.exists():
            pbar.update(len(tile_ids))
            continue

        ds = xr.open_dataset(nc_file)
        if ("time" not in ds.coords) or ("tp" not in ds.variables):
            ds.close()
            pbar.update(len(tile_ids))
            continue

        T_len = ds.sizes["time"]

        ds_year = ds.time.dt.year.values.astype(np.int32)
        ds_month = ds.time.dt.month.values.astype(np.int32)
        ds_day = ds.time.dt.day.values.astype(np.int32)
        ds_hour = ds.time.dt.hour.values.astype(np.int32)

        for tid in tile_ids:
            lat_coarse, lon_coarse, lat_fine, lon_fine = tile_coordinates(tid)

            coarse_ds = ds.tp.interp(lat=lat_coarse, lon=lon_coarse, method="linear")
            fine_ds = ds.tp.interp(lat=lat_fine, lon=lon_fine, method="linear")

            c_vals = coarse_ds.values  # shape (T, cLat, cLon)
            f_vals = fine_ds.values    # shape (T, fLat, fLon)

            # Write each time step individually so we can check zeros if needed.
            for i in range(T_len):
                c_slice = c_vals[i]
                if (not INCLUDE_ZEROS) and np.all(c_slice == 0.0):
                    # Skip if entire coarse grid is zero and we are excluding zeros
                    continue

                # Write to memory map
                input_mm[idx] = c_slice.astype(np.float32)
                target_mm[idx] = f_vals[i].astype(np.float32)

                time_mm[idx, 0] = ds_year[i]
                time_mm[idx, 1] = ds_month[i]
                time_mm[idx, 2] = ds_day[i]
                time_mm[idx, 3] = ds_hour[i]

                tile_mm[idx] = tid
                idx += 1

            pbar.update(1)

        ds.close()

    pbar.close()

    # Flush memmaps
    input_mm.flush()
    target_mm.flush()
    time_mm.flush()
    tile_mm.flush()

    print(f"\nTotal samples after processing (skipped zeros if INCLUDE_ZEROS=False): {idx}")

    # 5) Convert memmaps to arrays, train/test split
    input_arr = np.array(input_mm[:idx])
    target_arr = np.array(target_mm[:idx])
    time_arr = np.array(time_mm[:idx])
    tile_arr = np.array(tile_mm[:idx])

    data_dict = _split_data_by_date(
        date_array=time_arr,
        data_arrays=[input_arr, target_arr, time_arr, tile_arr],
        train_split=TRAIN_SPLIT
    )

    train_input, train_target, train_time, train_tile = data_dict['train']
    test_input,  test_target,  test_time,  test_tile  = data_dict['test']

    # Cast input/target arrays to SAVE_PRECISION
    train_input = train_input.astype(SAVE_PRECISION)
    train_target = train_target.astype(SAVE_PRECISION)
    test_input = test_input.astype(SAVE_PRECISION)
    test_target = test_target.astype(SAVE_PRECISION)

    print("Train samples:", len(train_input), "Test samples:", len(test_input))

    # 8) Load elevation data once, and create tile_elevations for each tile_id
    #    at the fine resolution.
    ds_elev = xr.open_dataset(ELEVATION_FILE)
    # If needed, rename dimensions from (Y, X) to (lat, lon).
    # This might vary depending on how the file is formatted; adjust if necessary:
    if 'Y' in ds_elev.dims and 'X' in ds_elev.dims:
        ds_elev = ds_elev.rename({'Y': 'lat', 'X': 'lon'})

    full_tile_dict = get_tile_dict()  # For all tiles, not just TRAINING_TILES
    # But we only need those in TRAINING_TILES. We'll store them in the same order as tile_ids
    unique_tile_ids = sorted(tile_ids)
    num_tiles = len(unique_tile_ids)
    tile_elevations = np.zeros((num_tiles, fLat, fLon), dtype=np.float32)

    for i, tid in enumerate(unique_tile_ids):
        _, _, lat_fine, lon_fine = tile_coordinates(tid)
        elev_vals = ds_elev.topo.interp(lat=lat_fine, lon=lon_fine, method='nearest').values/8848.9
        elev_vals = np.nan_to_num(elev_vals, nan=0.0)
        tile_elevations[i] = elev_vals.astype(np.float32)

    ds_elev.close()

    # We'll store tile_ids as a separate array
    tile_ids_arr = np.array(unique_tile_ids, dtype=np.int32)

    # 9) Save single .npz with train/test data plus tile elevation data
    out_combined_path = PROCESSED_DIR / "combined_data.npz"
    np.savez_compressed(
        out_combined_path,
        train_input=train_input,
        train_target=train_target,
        train_time=train_time,
        train_tile=train_tile,
        test_input=test_input,
        test_target=test_target,
        test_time=test_time,
        test_tile=test_tile,
        tile_elevations=tile_elevations,
        tile_ids=tile_ids_arr
    )

    print(f"\nSaved all data (train/test/elevation) to: {out_combined_path}")

    # 10) Remove temporary memmap files
    for p in [input_mm_path, target_mm_path, time_mm_path, tile_mm_path]:
        if p.exists():
            os.remove(p)
    print("Cleaned up temporary .dat files.")


if __name__ == "__main__":
    preprocess_data()
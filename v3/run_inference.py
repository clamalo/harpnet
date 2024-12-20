import os
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from src.tiles import get_all_tiles, tile_coordinates
from src.constants import (RAW_DIR, MODEL_NAME, TORCH_DEVICE,
                           MIN_LAT, MAX_LAT, MIN_LON, MAX_LON, MODEL_INPUT_CHANNELS,
                           MODEL_OUTPUT_CHANNELS, FIGURES_DIR, TILE_SIZE, FINE_RESOLUTION)
import importlib

# -----------------------------------------
# User-controlled variables
start_date = (1980, 9, 1)
end_date = (1980, 9, 3)
checkpoint_path = "/Users/clamalo/documents/harpnet/v3/checkpoints/best/best_model.pt"
plot_coarse = True  # If True, also display the coarse input field in the plots
# -----------------------------------------

OUTPUT_DIR = FIGURES_DIR / "inference_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MM_TO_INCHES = 0.0393701

model_module = importlib.import_module(f"src.models.{MODEL_NAME}")
ModelClass = model_module.Model

def load_model(checkpoint_path: str):
    """
    Load a trained model from a given checkpoint file.

    Args:
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        nn.Module: The loaded and evaluated model.
    """
    device = TORCH_DEVICE
    model = ModelClass().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def get_tile_elevation(tile: int, elevation_ds):
    """
    Retrieve and normalize elevation data for a specific tile.

    Args:
        tile (int): Tile index.
        elevation_ds (xarray.Dataset): Elevation dataset.

    Returns:
        np.ndarray: Normalized elevation array (1,Hf,Wf).
    """
    coarse_latitudes, coarse_longitudes, fine_latitudes, fine_longitudes = tile_coordinates(tile)
    elev_fine = elevation_ds.interp(lat=fine_latitudes, lon=fine_longitudes).topo.fillna(0.0).values.astype('float32')
    # Normalize by the approximate height of Everest
    elev_fine = elev_fine / 8848.9
    elev_fine = elev_fine[np.newaxis, ...]
    return elev_fine

def prepare_inputs_for_tile(tile: int, day_ds, elevation_ds):
    """
    Prepare the input tensors for a given tile on a given day.
    Interpolates coarse precipitation and adds elevation data.

    Args:
        tile (int): Tile index.
        day_ds (xarray.Dataset): Daily dataset of precipitation.
        elevation_ds (xarray.Dataset): Elevation dataset.

    Returns:
        times (np.ndarray): Array of timestamps.
        inputs (torch.Tensor): Model input tensor (T, C, 64, 64).
        coarse_tp_64_mm (np.ndarray): Coarse precipitation interpolated to 64x64 (in mm).
    """
    coarse_latitudes, coarse_longitudes, fine_latitudes, fine_longitudes = tile_coordinates(tile)
    # Interpolate coarse grid
    coarse_ds = day_ds.interp(lat=coarse_latitudes, lon=coarse_longitudes)
    times = coarse_ds.time.values
    T = len(times)

    # Convert coarse data and elevation to model inputs
    coarse_tp = coarse_ds.tp.values.astype('float32')
    coarse_tp = np.expand_dims(coarse_tp, axis=1)
    device = TORCH_DEVICE
    coarse_tp_torch = torch.from_numpy(coarse_tp).to(device)

    elev_fine = get_tile_elevation(tile, elevation_ds)
    elev_fine_torch = torch.from_numpy(elev_fine).to(device)

    # Resample coarse input to the target size and concatenate elevation
    coarse_tp_64 = torch.nn.functional.interpolate(coarse_tp_torch, size=(64,64), mode='nearest')
    elev_64 = elev_fine_torch.unsqueeze(0).expand(T, -1, -1, -1)
    inputs = torch.cat([coarse_tp_64, elev_64], dim=1)

    coarse_tp_64_mm = coarse_tp_64.squeeze(1).cpu().numpy()
    return times, inputs, coarse_tp_64_mm

def get_tile_ground_truth(tile: int, day_ds):
    """
    Extract the fine-resolution ground truth precipitation for a given tile.

    Args:
        tile (int): Tile index.
        day_ds (xarray.Dataset): Daily dataset of precipitation.

    Returns:
        np.ndarray: Fine-resolution precipitation (T,64,64) in mm.
    """
    _, _, fine_latitudes, fine_longitudes = tile_coordinates(tile)
    fine_ds = day_ds.interp(lat=fine_latitudes, lon=fine_longitudes)
    fine_tp = fine_ds.tp.values.astype('float32')
    return fine_tp

def stitch_tiles(tile_data_dict, tile_list):
    """
    Combine per-tile arrays into a single stitched image covering the entire domain.

    Args:
        tile_data_dict (dict): Dictionary {tile: np.ndarray(T,H,W)} to stitch.
        tile_list (list): List of tile indices.

    Returns:
        np.ndarray: Stitched array (T, H_full, W_full).
    """
    grid_domains = get_all_tiles()
    tile_size_degrees = TILE_SIZE * FINE_RESOLUTION
    lat_count = int((MAX_LAT - MIN_LAT) / tile_size_degrees)
    lon_count = int((MAX_LON - MIN_LON) / tile_size_degrees)

    # Determine T from the first tile
    T = None
    for t in tile_list:
        arr = tile_data_dict[t]
        T = arr.shape[0]
        break

    stitched = np.zeros((T, 64*lat_count, 64*lon_count), dtype='float32')

    # Place each tile in the correct position in the stitched output
    for tile in tile_list:
        lat_min, lat_max, lon_min, lon_max = grid_domains[tile]
        tile_row = int((lat_min - MIN_LAT) / tile_size_degrees)
        tile_col = int((lon_min - MIN_LON) / tile_size_degrees)

        arr = tile_data_dict[tile]
        row_start = tile_row * 64
        row_end = row_start + 64
        col_start = tile_col * 64
        col_end = col_start + 64

        stitched[:, row_start:row_end, col_start:col_end] = arr

    return stitched

def plot_and_save_images(stitched_coarse, stitched_model, stitched_truth, times, year, month, day, plot_coarse):
    """
    Plot and save daily precipitation maps (coarse, model output, ground truth).
    Handles either two-plot or three-plot layout depending on `plot_coarse`.

    Args:
        stitched_coarse (np.ndarray): Coarse interpolated precipitation in inches.
        stitched_model (np.ndarray): Model downscaled precipitation in inches.
        stitched_truth (np.ndarray): Ground truth fine-resolution precipitation in inches.
        times (np.ndarray): Array of timestamps.
        year, month, day (int): Date parameters for saving plots.
        plot_coarse (bool): If True, include coarse plot in the figure.
    """
    vmax_val = 0.5
    cmap_choice = 'viridis'
    extent = [MIN_LON, MAX_LON, MIN_LAT, MAX_LAT]

    for i, t in enumerate(times):
        if plot_coarse:
            fig = plt.figure(figsize=(12,6))
            # Coarse Input
            ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
            ax1.set_extent(extent, crs=ccrs.PlateCarree())
            ax1.coastlines()
            ax1.add_feature(cfeature.STATES, edgecolor='black')
            im1 = ax1.imshow(stitched_coarse[i], origin='lower', cmap=cmap_choice,
                              extent=extent, transform=ccrs.PlateCarree(),
                              vmin=0, vmax=vmax_val)
            ax1.set_title("Coarse Input")

            # Downscaled Output
            ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
            ax2.set_extent(extent, crs=ccrs.PlateCarree())
            ax2.coastlines()
            ax2.add_feature(cfeature.STATES, edgecolor='black')
            im2 = ax2.imshow(stitched_model[i], origin='lower', cmap=cmap_choice,
                              extent=extent, transform=ccrs.PlateCarree(),
                              vmin=0, vmax=vmax_val)
            ax2.set_title("Downscaled Output")

            # Ground Truth
            ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
            ax3.set_extent(extent, crs=ccrs.PlateCarree())
            ax3.coastlines()
            ax3.add_feature(cfeature.STATES, edgecolor='black')
            im3 = ax3.imshow(stitched_truth[i], origin='lower', cmap=cmap_choice,
                              extent=extent, transform=ccrs.PlateCarree(),
                              vmin=0, vmax=vmax_val)
            ax3.set_title("Ground Truth")

            fig.suptitle(f"Comparison - {str(t)}", fontsize=14)
            cbar = fig.colorbar(im1, ax=[ax1, ax2, ax3], fraction=0.046, pad=0.04)
            cbar.set_label('Precipitation (inches/3 hours)')
        else:
            # Only Model and Ground Truth
            fig = plt.figure(figsize=(12,6))
            ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
            ax1.set_extent(extent, crs=ccrs.PlateCarree())
            ax1.coastlines()
            ax1.add_feature(cfeature.STATES, edgecolor='black')
            im1 = ax1.imshow(stitched_model[i], origin='lower', cmap=cmap_choice,
                              extent=extent, transform=ccrs.PlateCarree(),
                              vmin=0, vmax=vmax_val)
            ax1.set_title("Downscaled Output")

            ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
            ax2.set_extent(extent, crs=ccrs.PlateCarree())
            ax2.coastlines()
            ax2.add_feature(cfeature.STATES, edgecolor='black')
            im2 = ax2.imshow(stitched_truth[i], origin='lower', cmap=cmap_choice,
                              extent=extent, transform=ccrs.PlateCarree(),
                              vmin=0, vmax=vmax_val)
            ax2.set_title("Ground Truth")

            fig.suptitle(f"Comparison - {str(t)}", fontsize=14)
            cbar = fig.colorbar(im1, ax=[ax1, ax2], fraction=0.046, pad=0.04)
            cbar.set_label('Precipitation (inches/3 hours)')

        filename = OUTPUT_DIR / f"{year}-{month:02d}-{day:02d}_{str(t).replace(':','-')}.png"
        plt.savefig(filename, dpi=150)
        plt.close(fig)

def plot_total_precipitation(total_coarse, total_model, total_truth, plot_coarse):
    """
    Plot the total accumulated precipitation over the entire processed period.
    Compares coarse, downscaled, and ground truth totals.

    Args:
        total_coarse (np.ndarray): Accumulated coarse precipitation in inches.
        total_model (np.ndarray): Accumulated model downscaled precipitation in inches.
        total_truth (np.ndarray): Accumulated ground truth precipitation in inches.
        plot_coarse (bool): If True, show coarse totals as well.
    """
    all_values = [total_model.max(), total_truth.max()]
    if plot_coarse:
        all_values.append(total_coarse.max())
    vmax_val = max(all_values)

    cmap_choice = 'viridis'
    extent = [MIN_LON, MAX_LON, MIN_LAT, MAX_LAT]

    if plot_coarse:
        fig = plt.figure(figsize=(12,6))
        # Coarse total
        ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
        ax1.set_extent(extent, crs=ccrs.PlateCarree())
        ax1.coastlines()
        ax1.add_feature(cfeature.STATES, edgecolor='black')
        im1 = ax1.imshow(total_coarse, origin='lower', cmap=cmap_choice,
                          extent=extent, transform=ccrs.PlateCarree(),
                          vmin=0, vmax=vmax_val)
        ax1.set_title("Coarse Total")

        # Downscaled total
        ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
        ax2.set_extent(extent, crs=ccrs.PlateCarree())
        ax2.coastlines()
        ax2.add_feature(cfeature.STATES, edgecolor='black')
        im2 = ax2.imshow(total_model, origin='lower', cmap=cmap_choice,
                          extent=extent, transform=ccrs.PlateCarree(),
                          vmin=0, vmax=vmax_val)
        ax2.set_title("Downscaled Total")

        # Ground truth total
        ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
        ax3.set_extent(extent, crs=ccrs.PlateCarree())
        ax3.coastlines()
        ax3.add_feature(cfeature.STATES, edgecolor='black')
        im3 = ax3.imshow(total_truth, origin='lower', cmap=cmap_choice,
                          extent=extent, transform=ccrs.PlateCarree(),
                          vmin=0, vmax=vmax_val)
        ax3.set_title("Ground Truth Total")

        fig.suptitle("Total Precipitation Over Entire Period", fontsize=14)
        cbar = fig.colorbar(im1, ax=[ax1, ax2, ax3], fraction=0.046, pad=0.04)
        cbar.set_label('Precipitation (inches)')
    else:
        # Just model and truth totals
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax1.set_extent(extent, crs=ccrs.PlateCarree())
        ax1.coastlines()
        ax1.add_feature(cfeature.STATES, edgecolor='black')
        im1 = ax1.imshow(total_model, origin='lower', cmap=cmap_choice,
                          extent=extent, transform=ccrs.PlateCarree(),
                          vmin=0, vmax=vmax_val)
        ax1.set_title("Downscaled Total")

        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        ax2.set_extent(extent, crs=ccrs.PlateCarree())
        ax2.coastlines()
        ax2.add_feature(cfeature.STATES, edgecolor='black')
        im2 = ax2.imshow(total_truth, origin='lower', cmap=cmap_choice,
                          extent=extent, transform=ccrs.PlateCarree(),
                          vmin=0, vmax=vmax_val)
        ax2.set_title("Ground Truth Total")

        fig.suptitle("Total Precipitation Over Entire Period", fontsize=14)
        cbar = fig.colorbar(im1, ax=[ax1, ax2], fraction=0.046, pad=0.04)
        cbar.set_label('Precipitation (inches)')

    filename = OUTPUT_DIR / "total_precipitation.png"
    plt.savefig(filename, dpi=150)
    plt.close(fig)

# Global accumulators for total precipitation
stitched_model_total = None
stitched_truth_total = None
stitched_coarse_total = None

def process_day(model, elevation_ds, day_ds, year, month, day):
    """
    Process one day of data for all tiles:
    - Performs model inference.
    - Stitches tile data into domain-wide arrays.
    - Accumulates daily totals and saves daily comparison plots.

    Args:
        model (nn.Module): The trained model.
        elevation_ds (xarray.Dataset): Elevation dataset.
        day_ds (xarray.Dataset): Dataset for the specific day.
        year, month, day (int): The date being processed.
    """
    global stitched_model_total, stitched_truth_total, stitched_coarse_total

    grid_domains = get_all_tiles()
    valid_tiles = list(grid_domains.keys())

    tile_predictions = {}
    tile_ground_truth_data = {}
    tile_coarse_data = {}
    times_reference = None

    # Run inference for each tile
    for tile in valid_tiles:
        times, inputs, coarse_tp_64_mm = prepare_inputs_for_tile(tile, day_ds, elevation_ds)
        if times_reference is None:
            times_reference = times
        else:
            # Ensure consistent time dimension across all tiles
            if not np.array_equal(times_reference, times):
                raise ValueError("Time dimension mismatch among tiles.")

        with torch.no_grad():
            preds = model(inputs)
            preds = preds.squeeze(1).cpu().numpy()

        fine_tp = get_tile_ground_truth(tile, day_ds)

        tile_predictions[tile] = preds
        tile_ground_truth_data[tile] = fine_tp
        tile_coarse_data[tile] = coarse_tp_64_mm

    # Stitch predictions, ground truth, and coarse data
    stitched_model = stitch_tiles(tile_predictions, valid_tiles)
    stitched_truth = stitch_tiles(tile_ground_truth_data, valid_tiles)
    stitched_coarse = stitch_tiles(tile_coarse_data, valid_tiles)

    # Convert mm to inches
    stitched_model_inches = stitched_model * MM_TO_INCHES
    stitched_truth_inches = stitched_truth * MM_TO_INCHES
    stitched_coarse_inches = stitched_coarse * MM_TO_INCHES

    # Plot daily comparisons
    plot_and_save_images(stitched_coarse_inches, stitched_model_inches, stitched_truth_inches,
                         times_reference, year, month, day, plot_coarse)
    print(f"Inference complete for {year}-{month:02d}-{day:02d}.")

    # Accumulate daily totals
    day_model_total = stitched_model_inches.sum(axis=0)
    day_truth_total = stitched_truth_inches.sum(axis=0)
    day_coarse_total = stitched_coarse_inches.sum(axis=0)

    if stitched_model_total is None:
        stitched_model_total = day_model_total.copy()
        stitched_truth_total = day_truth_total.copy()
        stitched_coarse_total = day_coarse_total.copy()
    else:
        stitched_model_total += day_model_total
        stitched_truth_total += day_truth_total
        stitched_coarse_total += day_coarse_total

def main():
    """
    Main function to run inference over a specified date range.
    For each day:
        - Load the model and elevation data.
        - Extract daily precipitation data.
        - Perform inference and plot outputs.
    Finally, plot the total accumulated precipitation over the entire period.
    """
    model = load_model(checkpoint_path)
    model.eval()

    elevation_ds = xr.open_dataset("/Users/clamalo/downloads/elevation.nc")
    # Rename dims if required
    if 'X' in elevation_ds.dims and 'Y' in elevation_ds.dims:
        elevation_ds = elevation_ds.rename({'X': 'lon', 'Y': 'lat'})

    start_dt = datetime(*start_date)
    end_dt = datetime(*end_date)
    current_dt = start_dt

    processed_any = False

    # Process each day in the range
    while current_dt <= end_dt:
        year, month, day = current_dt.year, current_dt.month, current_dt.day
        data_file = RAW_DIR / f"{year}-{month:02d}.nc"
        if not data_file.exists():
            print(f"Data file {data_file} not found. Skipping {year}-{month:02d}-{day:02d}.")
            current_dt += timedelta(days=1)
            continue

        month_ds = xr.open_dataset(data_file)
        day_str = f"{year}-{month:02d}-{day:02d}"
        start_time = np.datetime64(day_str)
        # Select the full day
        end_time = start_time + np.timedelta64(23, 'h') + np.timedelta64(59, 'm') + np.timedelta64(59, 's')
        day_slice = month_ds.sel(time=slice(start_time, end_time))
        if len(day_slice.time) == 0:
            print(f"No data found for {day_str} in {data_file}. Skipping.")
            current_dt += timedelta(days=1)
            continue

        # Process the day (run inference, save plots)
        process_day(model, elevation_ds, day_slice, year, month, day)
        processed_any = True
        current_dt += timedelta(days=1)

    # Once all days are processed, plot total precipitation if any data was processed
    if processed_any:
        plot_total_precipitation(stitched_coarse_total, stitched_model_total, stitched_truth_total, plot_coarse)
        print("Total precipitation plots saved.")

if __name__ == "__main__":
    main()
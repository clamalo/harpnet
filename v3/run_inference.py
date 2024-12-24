"""
Adapted inference script to use normalized inputs and then inverse normalization after inference.
If a tile-specific best model is found, it uses that; otherwise uses global best model.

Behavior on Missing Files:
    1. If the daily NetCDF file (e.g. year-month.nc) is not found, we log a warning and skip that day. 
       This allows partial coverage in the inference period.
    2. If no days were processed (i.e., all files were missing), no plots are generated.

All other logic for skipping or raising errors is identical to how the rest of the system handles data.
"""

import os
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import logging

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from src.tiles import get_all_tiles, tile_coordinates
from src.constants import (RAW_DIR, MODEL_NAME, TORCH_DEVICE,
                           MIN_LAT, MAX_LAT, MIN_LON, MAX_LON, FIGURES_DIR, TILE_SIZE, FINE_RESOLUTION, CHECKPOINTS_DIR, NORMALIZATION_STATS_FILE, TILE_SIZE, PRE_MODEL_INTERPOLATION)
import importlib

# -----------------------------------------
# User-controlled variables
start_date = (2019, 2, 13)
end_date = (2019, 2, 18)
global_best_checkpoint_path = CHECKPOINTS_DIR / "best" / "best_model.pt" 
plot_coarse = False  # If True, also display the coarse input field in the plots
# -----------------------------------------

OUTPUT_DIR = FIGURES_DIR / "inference_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MM_TO_INCHES = 0.0393701

model_module = importlib.import_module(f"src.models.{MODEL_NAME}")
ModelClass = model_module.Model

# Load normalization stats
norm_stats = np.load(NORMALIZATION_STATS_FILE)
mean_val, std_val = float(norm_stats[0]), float(norm_stats[1])

def load_model_for_tile(tile: int):
    device = TORCH_DEVICE
    tile_best_path = CHECKPOINTS_DIR / 'best' / f"{tile}_best.pt"
    if tile_best_path.exists():
        checkpoint_path = tile_best_path
    else:
        checkpoint_path = global_best_checkpoint_path

    model = ModelClass().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def get_tile_elevation(tile: int, elevation_ds):
    coarse_latitudes, coarse_longitudes, fine_latitudes, fine_longitudes = tile_coordinates(tile)
    elev_fine = elevation_ds.interp(lat=fine_latitudes, lon=fine_longitudes).topo.fillna(0.0).values.astype('float32')
    elev_fine = elev_fine / 8848.9
    elev_fine = elev_fine[np.newaxis, ...]
    return elev_fine

def prepare_inputs_for_tile(tile: int, day_ds, elevation_ds, model_device):
    coarse_latitudes, coarse_longitudes, fine_latitudes, fine_longitudes = tile_coordinates(tile)
    coarse_ds = day_ds.interp(lat=coarse_latitudes, lon=coarse_longitudes)
    times = coarse_ds.time.values
    T = len(times)

    coarse_tp = coarse_ds.tp.values.astype('float32')  # (T,Hc,Wc)
    if len(coarse_tp.shape) == 3:
        coarse_tp = coarse_tp[:, np.newaxis, :, :] # (T,1,Hc,Wc)
    coarse_tp_torch = torch.from_numpy(coarse_tp).to(model_device)

    elev_fine = get_tile_elevation(tile, elevation_ds)
    elev_fine_torch = torch.from_numpy(elev_fine).to(model_device)

    # Interpolate coarse to 64x64
    coarse_tp_64 = torch.nn.functional.interpolate(coarse_tp_torch, size=(TILE_SIZE,TILE_SIZE), mode=PRE_MODEL_INTERPOLATION)
    elev_64 = torch.nn.functional.interpolate(elev_fine_torch.unsqueeze(0).expand(T,-1,-1,-1), size=(TILE_SIZE,TILE_SIZE), mode=PRE_MODEL_INTERPOLATION)
    elev_64 = elev_64.squeeze(0)

    # Normalize precipitation
    coarse_tp_64_norm = (coarse_tp_64 - mean_val)/std_val

    inputs = torch.cat([coarse_tp_64_norm, elev_64], dim=1) # (T,2,64,64)
    coarse_tp_64_mm = coarse_tp_64.squeeze(1).cpu().numpy()  # For reference/plotting
    return times, inputs, coarse_tp_64_mm

def get_tile_ground_truth(tile: int, day_ds):
    _, _, fine_latitudes, fine_longitudes = tile_coordinates(tile)
    fine_ds = day_ds.interp(lat=fine_latitudes, lon=fine_longitudes)
    fine_tp = fine_ds.tp.values.astype('float32')
    return fine_tp  # shape: (T,Hf,Wf)

def stitch_tiles(tile_data_dict, tile_list):
    grid_domains = get_all_tiles()
    tile_size_degrees = TILE_SIZE * FINE_RESOLUTION
    lat_count = int((MAX_LAT - MIN_LAT) / tile_size_degrees)
    lon_count = int((MAX_LON - MIN_LON) / tile_size_degrees)

    T = None
    for t in tile_list:
        arr = tile_data_dict[t]
        T = arr.shape[0]
        break

    stitched = np.zeros((T, TILE_SIZE*lat_count, TILE_SIZE*lon_count), dtype='float32')

    for tile in tile_list:
        lat_min, lat_max, lon_min, lon_max = grid_domains[tile]
        tile_row = int((lat_min - MIN_LAT) / tile_size_degrees)
        tile_col = int((lon_min - MIN_LON) / tile_size_degrees)

        arr = tile_data_dict[tile]
        row_start = tile_row * TILE_SIZE
        row_end = row_start + TILE_SIZE
        col_start = tile_col * TILE_SIZE
        col_end = col_start + TILE_SIZE

        stitched[:, row_start:row_end, col_start:col_end] = arr

    return stitched

def plot_and_save_images(stitched_coarse, stitched_model, stitched_truth, times, year, month, day, plot_coarse):
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
        ax1.add_feature(cfeature.STATES)
        im1 = ax1.imshow(total_coarse, origin='lower', cmap=cmap_choice,
                          extent=extent, transform=ccrs.PlateCarree(),
                          vmin=0, vmax=vmax_val)
        ax1.set_title("Coarse Total")

        # Downscaled total
        ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
        ax2.set_extent(extent, crs=ccrs.PlateCarree())
        ax2.coastlines()
        ax2.add_feature(cfeature.STATES)
        im2 = ax2.imshow(total_model, origin='lower', cmap=cmap_choice,
                          extent=extent, transform=ccrs.PlateCarree(),
                          vmin=0, vmax=vmax_val)
        ax2.set_title("Downscaled Total")

        # Ground Truth total
        ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
        ax3.set_extent(extent, crs=ccrs.PlateCarree())
        ax3.coastlines()
        ax3.add_feature(cfeature.STATES)
        im3 = ax3.imshow(total_truth, origin='lower', cmap=cmap_choice,
                          extent=extent, transform=ccrs.PlateCarree(),
                          vmin=0, vmax=vmax_val)
        ax3.set_title("Ground Truth Total")

        fig.suptitle("Total Precipitation Over Entire Period", fontsize=14)
        cbar = fig.colorbar(im1, ax=[ax1, ax2, ax3], fraction=0.046, pad=0.04)
        cbar.set_label('Precipitation (inches)')
    else:
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax1.set_extent(extent, crs=ccrs.PlateCarree())
        ax1.coastlines()
        ax1.add_feature(cfeature.STATES)
        im1 = ax1.imshow(total_model, origin='lower', cmap=cmap_choice,
                          extent=extent, transform=ccrs.PlateCarree(),
                          vmin=0, vmax=vmax_val)
        ax1.set_title("Downscaled Total")

        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        ax2.set_extent(extent, crs=ccrs.PlateCarree())
        ax2.coastlines()
        ax2.add_feature(cfeature.STATES)
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

stitched_model_total = None
stitched_truth_total = None
stitched_coarse_total = None

def process_day(elevation_ds, day_slice, year, month, day):
    global stitched_model_total, stitched_truth_total, stitched_coarse_total

    grid_domains = get_all_tiles()
    valid_tiles = list(grid_domains.keys())

    tile_predictions = {}
    tile_ground_truth_data = {}
    tile_coarse_data = {}

    device = TORCH_DEVICE

    for tile in valid_tiles:
        model = load_model_for_tile(tile)
        times, inputs, coarse_tp_64_mm = prepare_inputs_for_tile(tile, day_slice, elevation_ds, device)

        with torch.no_grad():
            preds_norm = model(inputs)  # preds in normalized form
            preds_norm = preds_norm.squeeze(1).cpu().numpy()

        preds_mm = (preds_norm * std_val) + mean_val

        fine_tp = get_tile_ground_truth(tile, day_slice)
        fine_tp_torch = torch.from_numpy(fine_tp)
        fine_tp_torch = fine_tp_torch.unsqueeze(1)
        # Ground truth also resized with the same mode
        fine_tp_64 = torch.nn.functional.interpolate(fine_tp_torch, size=(TILE_SIZE,TILE_SIZE), mode=PRE_MODEL_INTERPOLATION).squeeze(1).numpy()

        tile_predictions[tile] = preds_mm
        tile_ground_truth_data[tile] = fine_tp_64
        tile_coarse_data[tile] = coarse_tp_64_mm

    stitched_model = stitch_tiles(tile_predictions, valid_tiles)
    stitched_truth = stitch_tiles(tile_ground_truth_data, valid_tiles)
    stitched_coarse = stitch_tiles(tile_coarse_data, valid_tiles)

    stitched_model_inches = stitched_model * MM_TO_INCHES
    stitched_truth_inches = stitched_truth * MM_TO_INCHES
    stitched_coarse_inches = stitched_coarse * MM_TO_INCHES

    plot_and_save_images(stitched_coarse_inches, stitched_model_inches, stitched_truth_inches,
                         times, year, month, day, plot_coarse)
    logging.info(f"Inference complete for {year}-{month:02d}-{day:02d}.")

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
    Runs inference over the specified date range. Any day whose data file doesn't exist 
    gets skipped with a warning. If no days are successfully processed, nothing is plotted.
    """
    elevation_ds = xr.open_dataset("/Users/clamalo/downloads/elevation.nc")
    if 'X' in elevation_ds.dims and 'Y' in elevation_ds.dims:
        elevation_ds = elevation_ds.rename({'X': 'lon', 'Y': 'lat'})

    start_dt = datetime(*start_date)
    end_dt = datetime(*end_date)
    current_dt = start_dt

    processed_any = False

    while current_dt <= end_dt:
        year, month, day = current_dt.year, current_dt.month, current_dt.day
        data_file = RAW_DIR / f"{year}-{month:02d}.nc"
        if not data_file.exists():
            logging.warning(f"Data file {data_file} not found. Skipping {year}-{month:02d}-{day:02d}.")
            current_dt += timedelta(days=1)
            continue

        month_ds = xr.open_dataset(data_file)
        day_str = f"{year}-{month:02d}-{day:02d}"
        start_time = np.datetime64(day_str)
        end_time = start_time + np.timedelta64(23, 'h') + np.timedelta64(59, 'm') + np.timedelta64(59, 's')
        day_slice = month_ds.sel(time=slice(start_time, end_time))
        if len(day_slice.time) == 0:
            logging.warning(f"No data found for {day_str} in {data_file}. Skipping.")
            current_dt += timedelta(days=1)
            continue

        process_day(elevation_ds, day_slice, year, month, day)
        processed_any = True
        current_dt += timedelta(days=1)

    if processed_any:
        plot_total_precipitation(stitched_coarse_total, stitched_model_total, stitched_truth_total, plot_coarse)
        logging.info("Total precipitation plots saved.")

if __name__ == "__main__":
    main()
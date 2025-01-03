import importlib
import torch
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm
from datetime import date, timedelta

from config import (
    RAW_DIR,
    ELEVATION_FILE,
    MODEL_NAME,
    DEVICE,
    FIGURES_DIR,
    PROCESSED_DIR,
    INCLUDE_ZEROS,
    CHECKPOINTS_DIR,
    FINE_RESOLUTION,
    COARSE_RESOLUTION,
    PRECIP_THRESHOLD
)
from tiles import (
    get_tile_dict,
    tile_coordinates
)


def _date_range(start_day=(1980, 9, 3), end_day=(1980, 9, 5)):
    """
    Generate (year, month, day) tuples for each date from start_day up to and including end_day.
    """
    sy, sm, sd = start_day
    ey, em, ed = end_day
    start_dt = date(sy, sm, sd)
    end_dt = date(ey, em, ed)
    delta = end_dt - start_dt
    for i in range(delta.days + 1):
        d = start_dt + timedelta(days=i)
        yield (d.year, d.month, d.day)


def _upsample_coarse_with_crop_torch(coarse_tensor: torch.Tensor, final_shape: tuple) -> torch.Tensor:
    """
    Similar to generate_dataloaders.py, but works directly on a torch.Tensor.

    coarse_tensor: shape (1, 1, cLat, cLon), log-space precipitation
    final_shape: (fLat, fLon)
    Returns: shape (1, 1, fLat, fLon)
    """
    _, _, cLat, cLon = coarse_tensor.shape
    fLat, fLon = final_shape

    ratio = int(round(COARSE_RESOLUTION / FINE_RESOLUTION))
    upsample_size_lat = cLat * ratio
    upsample_size_lon = cLon * ratio

    # Interpolate
    upsampled_t = F.interpolate(
        coarse_tensor,
        size=(upsample_size_lat, upsample_size_lon),
        mode='bilinear',
        align_corners=False
    )

    # Center crop
    lat_diff = upsample_size_lat - fLat
    lon_diff = upsample_size_lon - fLon
    top = lat_diff // 2
    left = lon_diff // 2
    bottom = top + fLat
    right = left + fLon

    return upsampled_t[:, :, top:bottom, left:right]


def run_inference(
    start_day=(1980, 9, 3),
    end_day=(1980, 9, 3),
    checkpoint_path=None
):
    """
    Run the model inference for all dates on and between start_day and end_day,
    across all *primary* tiles in get_tile_dict(). This script will:

    1) Load the specified model checkpoint (general weights).
    2) Check whether there is a *tile-specific* fine-tuned model in CHECKPOINTS_DIR/best/
       for each tile. If so, use that tile’s fine-tuned weights instead of the general one.
    3) Load precipitation normalization stats (mean, std) from combined_data.npz.
    4) For each date in [start_day, end_day], and for each primary tile:
       - Open the NetCDF file for that month/year if it exists in RAW_DIR.
       - Slice out the day of interest at coarse resolution (0.25).
       - Keep precipitation in mm, apply log(1 + x), then normalize using the dataset-wide
         mean/std for the logged data.
       - **Now upsample to a bigger padded shape, then center-crop** so the final shape
         matches the 64×64 tile domain exactly.
       - Add the corresponding fine-resolution elevation channel to form a 2-channel input.
       - Run the model to get a (tile_size, tile_size) prediction for each hour.
       - Also retrieve the ground truth high-resolution data in mm (but do NOT log or normalize).
    5) Stitch the predictions for each tile into a single cohesive domain array, and do the same
       for ground truth, for each hour. Then un-normalize by reversing the log transform (expm1).
    6) For each hour, plot a 2-panel figure named 'inference_YYYYMMDD_HH.png'.
    7) After processing all days, plot a single 2-panel figure showing the *total* model
       precipitation vs. *total* ground truth across the entire date range.
    """

    def _fmt_date(yr, mn, dy):
        return f"{yr:04d}{mn:02d}{dy:02d}"

    # ---------------------------------------------------------------------
    # 1) Load the general model checkpoint, set device
    # ---------------------------------------------------------------------
    if DEVICE.lower() == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif DEVICE.lower() == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            print("Warning: MPS requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model_module = importlib.import_module(MODEL_NAME)
    ModelClass = model_module.Model

    # Load the general checkpoint
    general_ckpt_data = torch.load(checkpoint_path, map_location=device)
    general_ckpt_state_dict = general_ckpt_data.get('model_state_dict', None)
    if general_ckpt_state_dict is None:
        raise ValueError(f"Could not load 'model_state_dict' from {checkpoint_path}")

    # Directory for tile-specific best weights
    tile_best_dir = CHECKPOINTS_DIR / "best"

    # Create a dictionary mapping tile_id -> model
    tile_dict = get_tile_dict()
    primary_tile_dict = {tid: tile for tid, tile in tile_dict.items() if tile[-1] == 'primary'}
    tile_ids = sorted(primary_tile_dict.keys())

    tile_models = {}
    for tid in tile_ids:
        tile_model = ModelClass().to(device)
        tile_specific_ckpt = tile_best_dir / f"{tid}_best.pt"

        if tile_specific_ckpt.exists():
            print(f"Found fine-tuned weights for tile {tid} at {tile_specific_ckpt}")
            ckpt_data = torch.load(tile_specific_ckpt, map_location=device)
            tile_state_dict = ckpt_data.get('model_state_dict', None)
            if tile_state_dict is None:
                print(f"Warning: 'model_state_dict' missing in {tile_specific_ckpt}. Using general weights.")
                tile_model.load_state_dict(general_ckpt_state_dict, strict=True)
            else:
                tile_model.load_state_dict(tile_state_dict, strict=True)
        else:
            tile_model.load_state_dict(general_ckpt_state_dict, strict=True)

        tile_model.eval()
        tile_models[tid] = tile_model

    # ---------------------------------------------------------------------
    # 3) Load precipitation normalization stats (mean, std) for log(mm)
    # ---------------------------------------------------------------------
    data_path = PROCESSED_DIR / "combined_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cannot find {data_path}. Make sure you've run data_preprocessing.py first."
        )
    norm_data = np.load(data_path)
    precip_mean = norm_data["precip_mean"].item()
    precip_std = norm_data["precip_std"].item()
    norm_data.close()

    print(f"Loaded normalization stats (log scale): mean={precip_mean:.4f}, std={precip_std:.4f}")

    # ---------------------------------------------------------------------
    # Prepare global domain bounding box & lat/lon
    # ---------------------------------------------------------------------
    all_lats_min = []
    all_lats_max = []
    all_lons_min = []
    all_lons_max = []
    for tid in tile_ids:
        min_lat, max_lat, min_lon, max_lon, _ = primary_tile_dict[tid]
        all_lats_min.append(min_lat)
        all_lats_max.append(max_lat)
        all_lons_min.append(min_lon)
        all_lons_max.append(max_lon)

    min_lat_domain = min(all_lats_min)
    max_lat_domain = max(all_lats_max)
    min_lon_domain = min(all_lons_min)
    max_lon_domain = max(all_lons_max)

    # Fine resolution is typically 0.125 or 0.0625, etc.
    lat_global = np.arange(min_lat_domain, max_lat_domain, FINE_RESOLUTION)
    lon_global = np.arange(min_lon_domain, max_lon_domain, FINE_RESOLUTION)
    nLat = len(lat_global)
    nLon = len(lon_global)

    # We'll accumulate total precipitation across all days
    domain_pred_sum = np.zeros((nLat, nLon), dtype=np.float32)
    domain_true_sum = np.zeros((nLat, nLon), dtype=np.float32)

    # Load the elevation data once
    ds_elev = xr.open_dataset(ELEVATION_FILE)
    if 'Y' in ds_elev.dims and 'X' in ds_elev.dims:
        ds_elev = ds_elev.rename({'Y': 'lat', 'X': 'lon'})

    tile_elevations = {}
    for tid in tile_ids:
        _, _, lat_fine, lon_fine = tile_coordinates(tid)
        elev_vals = ds_elev.topo.interp(lat=lat_fine, lon=lon_fine, method='nearest').values / 8848.9
        elev_vals = np.nan_to_num(elev_vals, nan=0.0).astype(np.float32)
        tile_elevations[tid] = elev_vals
    ds_elev.close()

    out_dir = FIGURES_DIR / "inference_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    map_extent = [min_lon_domain, max_lon_domain, min_lat_domain, max_lat_domain]

    # ---------------------------------------------------------------------
    # 4) Iterate over each day in the range, run inference tile by tile
    # ---------------------------------------------------------------------
    for (year, month, day) in _date_range(start_day, end_day):
        day_str = f"{year:04d}-{month:02d}-{day:02d}"
        day_tag = _fmt_date(year, month, day)

        # For each day, store up to 24 hours
        domain_pred_day = np.full((24, nLat, nLon), np.nan, dtype=np.float32)
        domain_true_day = np.full((24, nLat, nLon), np.nan, dtype=np.float32)

        nc_file = RAW_DIR / f"{year:04d}-{month:02d}.nc"
        if not nc_file.exists():
            print(f"** Skipping {day_str}: No data file {nc_file}")
            continue

        ds = xr.open_dataset(nc_file)
        if ("time" not in ds.coords) or ("tp" not in ds.variables):
            ds.close()
            print(f"** Skipping {day_str}: 'tp' variable or 'time' coord not found")
            continue

        ds_day = ds.sel(time=day_str)
        if ds_day.time.size == 0:
            ds.close()
            print(f"** Skipping {day_str}: No data found for day in {nc_file}")
            continue

        ds_hour = ds_day.time.dt.hour.values.astype(int)
        hour_to_index = {}
        for i, h in enumerate(ds_hour):
            hour_to_index[h] = i

        print(f"Running inference for {day_str} across {len(tile_ids)} primary tiles...")

        for tid in tqdm(tile_ids, desc=f"Processing {day_str}"):
            min_lat, max_lat, min_lon, max_lon, _ = primary_tile_dict[tid]
            lat_coarse, lon_coarse, lat_fine, lon_fine = tile_coordinates(tid)

            model = tile_models[tid]

            # Interpolate dataset onto tile's coarse grid
            coarse_ds = ds_day.tp.interp(lat=lat_coarse, lon=lon_coarse, method="linear")
            fine_ds = ds_day.tp.interp(lat=lat_fine, lon=lon_fine, method="linear")

            c_vals = coarse_ds.values  # shape (T, cLat, cLon)
            f_vals = fine_ds.values    # shape (T, fLat, fLon)
            elev_tile = tile_elevations[tid]

            for i_hr, h in enumerate(ds_hour):
                coarse_precip_mm = c_vals[i_hr].astype(np.float32)
                if (not INCLUDE_ZEROS) and (coarse_precip_mm.max() < PRECIP_THRESHOLD):
                    # If skipping zeros
                    pred_arr = np.zeros_like(elev_tile, dtype=np.float32)
                else:
                    # log(1 + x), then normalize
                    coarse_precip_log = np.log1p(coarse_precip_mm)
                    coarse_precip_norm = (coarse_precip_log - precip_mean) / precip_std

                    # Upsample to bigger padded shape -> crop to tile_size
                    coarse_tensor = torch.from_numpy(coarse_precip_norm).unsqueeze(0).unsqueeze(0).to(device)
                    upsampled_cropped_t = _upsample_coarse_with_crop_torch(
                        coarse_tensor,
                        final_shape=elev_tile.shape  # e.g. (64,64)
                    )

                    # Elevation channel
                    elev_tile_t = torch.from_numpy(elev_tile).unsqueeze(0).unsqueeze(0).to(device)

                    model_input = torch.cat([upsampled_cropped_t, elev_tile_t], dim=1)
                    with torch.no_grad():
                        pred_t = model(model_input)
                    pred_arr_norm = pred_t.squeeze().cpu().numpy()

                    # Un-normalize
                    pred_arr_log = (pred_arr_norm * precip_std) + precip_mean
                    pred_arr = np.expm1(pred_arr_log)

                # Place it into the domain
                hour_idx = h if h in hour_to_index else None
                if hour_idx is not None:
                    lat_indices = np.searchsorted(lat_global, lat_fine)
                    lon_indices = np.searchsorted(lon_global, lon_fine)

                    domain_pred_day[hour_idx,
                                    lat_indices[0]:lat_indices[-1] + 1,
                                    lon_indices[0]:lon_indices[-1] + 1] = pred_arr

                    # Ground truth
                    domain_true_day[hour_idx,
                                    lat_indices[0]:lat_indices[-1] + 1,
                                    lon_indices[0]:lon_indices[-1] + 1] = f_vals[i_hr]

        ds.close()

        # Plot each hour
        plotted_hours = sorted(hour_to_index.keys())
        for h in plotted_hours:
            hour_idx = h
            pred_2d = domain_pred_day[hour_idx]
            true_2d = domain_true_day[hour_idx]

            if np.all(np.isnan(pred_2d)):
                continue

            fig, axes = plt.subplots(
                1, 2,
                subplot_kw={'projection': ccrs.PlateCarree()},
                figsize=(12, 6)
            )
            ax_pred, ax_true = axes

            for ax in axes:
                ax.set_extent(map_extent, crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=1)
                ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
                ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)

            im_pred = ax_pred.imshow(
                pred_2d,
                origin='lower',
                extent=map_extent,
                transform=ccrs.PlateCarree(),
                cmap='viridis',
                vmin=0.0,
                vmax=10.0
            )
            ax_pred.set_title(f"Downscaled Model Output\n{day_str} {hour_idx:02d} UTC")

            im_true = ax_true.imshow(
                true_2d,
                origin='lower',
                extent=map_extent,
                transform=ccrs.PlateCarree(),
                cmap='viridis',
                vmin=0.0,
                vmax=10.0
            )
            ax_true.set_title(f"Ground Truth\n{day_str} {hour_idx:02d} UTC")

            fig.subplots_adjust(bottom=0.15)
            cbar = fig.colorbar(im_true, ax=axes, orientation='horizontal', fraction=0.046, pad=0.08)
            cbar.set_label("Precip (mm / 3hr)")

            out_fname = out_dir / f"inference_{day_tag}_{hour_idx:02d}.png"
            plt.savefig(out_fname, dpi=150, bbox_inches='tight')
            plt.close(fig)

        # Sum up all valid hours for this day
        if len(plotted_hours) > 0:
            day_pred_sum = np.nansum(domain_pred_day[plotted_hours], axis=0)
            day_true_sum = np.nansum(domain_true_day[plotted_hours], axis=0)
            domain_pred_sum += np.nan_to_num(day_pred_sum)
            domain_true_sum += np.nan_to_num(day_true_sum)

    # ---------------------------------------------------------------------
    # 7) After processing all days, plot total sum across entire date range
    # ---------------------------------------------------------------------
    total_max = max(np.nanmax(domain_pred_sum), np.nanmax(domain_true_sum))
    if total_max < 0 or np.isnan(total_max):
        print("No valid data across the entire date range.")
        return

    fig, axes = plt.subplots(
        1, 2,
        subplot_kw={'projection': ccrs.PlateCarree()},
        figsize=(12, 6)
    )
    ax_pred_sum, ax_true_sum = axes
    for ax in axes:
        ax.set_extent(map_extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
        ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)

    im_pred_sum = ax_pred_sum.imshow(
        domain_pred_sum,
        origin='lower',
        extent=map_extent,
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        vmin=0.0,
        vmax=total_max if total_max > 0 else 10.0
    )
    ax_pred_sum.set_title(f"Total Model Output\n({_fmt_date(*start_day)} - {_fmt_date(*end_day)})")

    im_true_sum = ax_true_sum.imshow(
        domain_true_sum,
        origin='lower',
        extent=map_extent,
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        vmin=0.0,
        vmax=total_max if total_max > 0 else 10.0
    )
    ax_true_sum.set_title(f"Total Ground Truth\n({_fmt_date(*start_day)} - {_fmt_date(*end_day)})")

    fig.subplots_adjust(bottom=0.15)
    cbar2 = fig.colorbar(im_true_sum, ax=axes, orientation='horizontal', fraction=0.046, pad=0.08)
    cbar2.set_label("Total Precip (mm)")

    out_total = out_dir / f"inference_total_{_fmt_date(*start_day)}_{_fmt_date(*end_day)}.png"
    plt.savefig(out_total, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nAll inference plots saved to: {out_dir}")
    print("Done!")
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
    TILE_SIZE,
    SECONDARY_TILES
)
from tiles import (
    get_tile_dict,
    tile_coordinates
)


def tile_weight_mask(N=TILE_SIZE):
    """
    Create an N×N NumPy array of Manhattan-based weights.
    The four central cells are ~1, and edges approach 0.
    """
    if N % 2 != 0:
        raise ValueError("N must be an even number.")

    # Indices of the 2×2 center
    c1 = N // 2 - 1
    c2 = N // 2

    # Grid of row/col indices
    rows, cols = np.indices((N, N))

    # Compute Manhattan distance to each of the 4 central cells
    dist1 = np.abs(rows - c1) + np.abs(cols - c1)
    dist2 = np.abs(rows - c1) + np.abs(cols - c2)
    dist3 = np.abs(rows - c2) + np.abs(cols - c1)
    dist4 = np.abs(rows - c2) + np.abs(cols - c2)

    # Minimum distance to any center cell
    dist = np.minimum(np.minimum(dist1, dist2), np.minimum(dist3, dist4))
    max_dist = dist.max()

    # Linear flip so distance=0 => weight=1, distance=max => weight=0
    mask = 1.0 - (dist / max_dist) if max_dist > 0 else np.ones((N, N), dtype=np.float32)
    return mask.astype(np.float32)


def run_inference_manhattan(
    start_day=(1980, 9, 3),
    end_day=(1980, 9, 3),
    checkpoint_path=None
):
    """
    Run model inference for [start_day..end_day], using a staggered tiling approach 
    for primary + secondary tiles (Manhattan weighting). The ground truth is stored 
    unchanged, so it never gets NaN-patches even if the model side is zero-skipped.

    Steps:
    1) Load the specified model checkpoint (general weights) with map_location=DEVICE.
    2) For each tile (both primary and secondary), check for a fine-tuned
       checkpoint in CHECKPOINTS_DIR/best/<tile_id>_best.pt. If found, use that;
       otherwise use the general weights. Move the model to DEVICE.
    3) Load precipitation normalization stats (mean, std) from combined_data.npz.
    4) For each date in [start_day, end_day]:
       - Open the NetCDF file for that month/year if it exists.
       - For each tile & each hour of the day:
         (a) Fill ground truth domain_true_day. 
         (b) If INCLUDE_ZEROS=False and coarse is all zeros, skip model inference. 
             Otherwise, infer and multiply by the Manhattan mask.
       - After all tiles, finalize the mosaic by dividing domain_pred_day by domain_wt_day 
         wherever domain_wt_day > 0, then un-transform from log space to mm.
       - Plot side-by-side with ground truth, using the same figure names and directory 
         as run_inference.py.
    5) After all days, plot total model precipitation vs. total ground truth in the same manner.
    """
    def _fmt_date(yr, mn, dy):
        return f"{yr:04d}{mn:02d}{dy:02d}"

    # ---------------------------------------------------------------------
    # Basic checks for checkpoint
    # ---------------------------------------------------------------------
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # ---------------------------------------------------------------------
    # 1) Load the general model checkpoint
    # ---------------------------------------------------------------------
    model_module = importlib.import_module(MODEL_NAME)
    ModelClass = model_module.Model

    general_ckpt_data = torch.load(checkpoint_path, map_location=DEVICE)
    general_ckpt_state_dict = general_ckpt_data.get('model_state_dict', None)
    if general_ckpt_state_dict is None:
        raise ValueError(f"Could not load 'model_state_dict' from {checkpoint_path}")

    # ---------------------------------------------------------------------
    # 2) Build tile models from best checkpoints if available
    # ---------------------------------------------------------------------
    tile_dict = get_tile_dict()
    if not tile_dict:
        raise RuntimeError("No tiles found in get_tile_dict(). Check your config settings.")

    tile_ids = sorted(tile_dict.keys())
    tile_models = {}

    for tid in tile_ids:
        tile_model = ModelClass().to(DEVICE)
        tile_specific_ckpt = CHECKPOINTS_DIR / "best" / f"{tid}_best.pt"

        if tile_specific_ckpt.exists():
            print(f"Tile {tid}: using fine-tuned weights {tile_specific_ckpt}")
            ckpt_data = torch.load(tile_specific_ckpt, map_location=DEVICE)
            tile_state_dict = ckpt_data.get('model_state_dict', None)
            if tile_state_dict is not None:
                tile_model.load_state_dict(tile_state_dict, strict=True)
            else:
                print(f"Warning: 'model_state_dict' missing in {tile_specific_ckpt}. Using general weights.")
                tile_model.load_state_dict(general_ckpt_state_dict, strict=True)
        else:
            tile_model.load_state_dict(general_ckpt_state_dict, strict=True)

        tile_model.eval()
        tile_models[tid] = tile_model

    # ---------------------------------------------------------------------
    # 3) Load normalization stats (mean, std)
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

    print(f"Normalization stats: mean={precip_mean:.4f}, std={precip_std:.4f}")

    # ---------------------------------------------------------------------
    # 4) Prepare global domain bounding box & arrays
    # ---------------------------------------------------------------------
    all_lats_min = [tile_dict[tid][0] for tid in tile_ids]
    all_lats_max = [tile_dict[tid][1] for tid in tile_ids]
    all_lons_min = [tile_dict[tid][2] for tid in tile_ids]
    all_lons_max = [tile_dict[tid][3] for tid in tile_ids]

    min_lat_domain = min(all_lats_min)
    max_lat_domain = max(all_lats_max)
    min_lon_domain = min(all_lons_min)
    max_lon_domain = max(all_lons_max)

    lat_global = np.arange(min_lat_domain, max_lat_domain, FINE_RESOLUTION, dtype=np.float32)
    lon_global = np.arange(min_lon_domain, max_lon_domain, FINE_RESOLUTION, dtype=np.float32)
    nLat = len(lat_global)
    nLon = len(lon_global)
    map_extent = [min_lon_domain, max_lon_domain, min_lat_domain, max_lat_domain]

    # For total precipitation accumulation
    domain_pred_sum = np.zeros((nLat, nLon), dtype=np.float32)
    domain_true_sum = np.zeros((nLat, nLon), dtype=np.float32)

    # ---------------------------------------------------------------------
    # Load tile elevations
    # ---------------------------------------------------------------------
    ds_elev = xr.open_dataset(ELEVATION_FILE)
    if 'Y' in ds_elev.dims and 'X' in ds_elev.dims:
        ds_elev = ds_elev.rename({'Y': 'lat', 'X': 'lon'})

    tile_elevations = {}
    for tid in tile_ids:
        _, _, lat_fine, lon_fine = tile_coordinates(tid)
        elev_vals = ds_elev.topo.interp(lat=lat_fine, lon=lon_fine, method='nearest').values
        elev_vals = np.nan_to_num(elev_vals, nan=0.0).astype(np.float32)
        tile_elevations[tid] = elev_vals / 8848.9  # same norm as run_inference
    ds_elev.close()

    # Precompute the Manhattan mask
    manhattan_mask = tile_weight_mask(TILE_SIZE)

    # ---------------------------------------------------------------------
    # Use the same directory & filenames as run_inference.py
    # ---------------------------------------------------------------------
    out_dir = FIGURES_DIR / "inference_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 5) Iterate over each day, fill domain arrays & mosaic
    # ---------------------------------------------------------------------
    def _date_range(sd=(1980, 9, 3), ed=(1980, 9, 5)):
        sy, sm, sd_ = sd
        ey, em, ed_ = ed
        start_dt = date(sy, sm, sd_)
        end_dt = date(ey, em, ed_)
        delta_ = (end_dt - start_dt).days
        for i in range(delta_ + 1):
            d_ = start_dt + timedelta(days=i)
            yield (d_.year, d_.month, d_.day)

    for (year, month, day) in _date_range(start_day, end_day):
        day_str = f"{year:04d}-{month:02d}-{day:02d}"
        day_tag = _fmt_date(year, month, day)

        # We'll hold data for 24 hours
        domain_pred_day = np.zeros((24, nLat, nLon), dtype=np.float32)
        domain_wt_day   = np.zeros((24, nLat, nLon), dtype=np.float32)
        domain_true_day = np.full((24, nLat, nLon), np.nan, dtype=np.float32)

        # Attempt reading the monthly file
        nc_file = RAW_DIR / f"{year:04d}-{month:02d}.nc"
        if not nc_file.exists():
            print(f"** Skipping {day_str}: No data file {nc_file}")
            continue

        ds = xr.open_dataset(nc_file)
        if ("time" not in ds.coords) or ("tp" not in ds.variables):
            ds.close()
            print(f"** Skipping {day_str}: 'tp' variable or 'time' coord not found.")
            continue

        ds_day = ds.sel(time=day_str)
        if ds_day.time.size == 0:
            ds.close()
            print(f"** Skipping {day_str}: No data found for {day_str} in {nc_file}")
            continue

        ds_hours = ds_day.time.dt.hour.values.astype(int)
        hour_to_index = {h: i for i, h in enumerate(ds_hours)}

        print(f"Manhattan inference for {day_str}, {len(tile_ids)} tile(s) total...")

        for tid in tqdm(tile_ids, desc=f"Processing {day_str}"):
            min_lat, max_lat, min_lon, max_lon, _ = tile_dict[tid]
            lat_coarse, lon_coarse, lat_fine, lon_fine = tile_coordinates(tid)

            tile_model = tile_models[tid]

            # Slice coarse & fine data
            coarse_ds = ds_day.tp.interp(lat=lat_coarse, lon=lon_coarse, method="linear")
            fine_ds   = ds_day.tp.interp(lat=lat_fine,  lon=lon_fine,  method="linear")

            c_vals = coarse_ds.values  # shape (T, cLat, cLon)
            f_vals = fine_ds.values    # shape (T, fLat, fLon)
            elev_tile = tile_elevations[tid]

            for i_hour, h_val in enumerate(ds_hours):
                hour_idx = h_val  # 0..23
                lat_indices = np.searchsorted(lat_global, lat_fine)
                lon_indices = np.searchsorted(lon_global, lon_fine)

                # 1) Always fill ground truth
                domain_true_day[hour_idx,
                                lat_indices[0]:lat_indices[-1]+1,
                                lon_indices[0]:lon_indices[-1]+1] = f_vals[i_hour]

                # 2) Possibly skip model inference if coarse is all zero
                coarse_precip_mm = c_vals[i_hour].astype(np.float32)
                if (not INCLUDE_ZEROS) and np.all(coarse_precip_mm == 0.0):
                    continue

                # Otherwise, do log/normalize -> model inference -> multiply by Manhattan mask
                coarse_precip_log  = np.log1p(coarse_precip_mm)
                coarse_precip_norm = (coarse_precip_log - precip_mean) / precip_std

                coarse_precip_t = torch.from_numpy(coarse_precip_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
                fLat, fLon = elev_tile.shape
                upsampled_precip_t = F.interpolate(
                    coarse_precip_t,
                    size=(fLat, fLon),
                    mode='bilinear',
                    align_corners=False
                )

                elev_tile_t = torch.from_numpy(elev_tile).unsqueeze(0).unsqueeze(0).to(DEVICE)
                model_input = torch.cat([upsampled_precip_t, elev_tile_t], dim=1)

                with torch.no_grad():
                    pred_t = tile_model(model_input)
                pred_arr_norm = pred_t.squeeze().cpu().numpy()

                domain_pred_day[hour_idx,
                                lat_indices[0]:lat_indices[-1]+1,
                                lon_indices[0]:lon_indices[-1]+1] += (pred_arr_norm * manhattan_mask)
                domain_wt_day[hour_idx,
                              lat_indices[0]:lat_indices[-1]+1,
                              lon_indices[0]:lon_indices[-1]+1] += manhattan_mask

        ds.close()

        # -----------------------------------------------------------------
        # Finalize mosaic & plot for each hour
        # -----------------------------------------------------------------
        plotted_hours = sorted(hour_to_index.keys())
        for h in plotted_hours:
            hour_idx = h

            with np.errstate(divide='ignore', invalid='ignore'):
                mosaic_norm = domain_pred_day[hour_idx] / domain_wt_day[hour_idx]
            mosaic_norm[~np.isfinite(mosaic_norm)] = 0.0

            # Un-normalize
            mosaic_log = (mosaic_norm * precip_std) + precip_mean
            mosaic_mm  = np.expm1(mosaic_log)

            # Ground truth
            mosaic_true = domain_true_day[hour_idx]

            if np.all(np.isnan(mosaic_true)) and np.all(mosaic_mm == 0):
                continue

            fig, (ax_pred, ax_true) = plt.subplots(
                1, 2,
                figsize=(12, 6),
                subplot_kw={'projection': ccrs.PlateCarree()}
            )
            for ax_ in (ax_pred, ax_true):
                ax_.set_extent(map_extent, crs=ccrs.PlateCarree())
                ax_.add_feature(cfeature.STATES, edgecolor='black', linewidth=1)
                ax_.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
                ax_.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)

            im_pred = ax_pred.imshow(
                mosaic_mm,
                origin='lower',
                extent=map_extent,
                transform=ccrs.PlateCarree(),
                cmap='viridis',
                vmin=0.0,
                vmax=10.0
            )
            ax_pred.set_title(f"Downscaled Model Output\n{day_str} {hour_idx:02d} UTC")

            im_true = ax_true.imshow(
                mosaic_true,
                origin='lower',
                extent=map_extent,
                transform=ccrs.PlateCarree(),
                cmap='viridis',
                vmin=0.0,
                vmax=10.0
            )
            ax_true.set_title(f"Ground Truth\n{day_str} {hour_idx:02d} UTC")

            fig.subplots_adjust(bottom=0.15)
            cbar = fig.colorbar(im_true, ax=(ax_pred, ax_true), orientation='horizontal', fraction=0.046, pad=0.08)
            cbar.set_label("Precip (mm / 3hr)")

            # Same filenames as run_inference.py
            out_fname = out_dir / f"inference_{day_tag}_{hour_idx:02d}.png"
            plt.savefig(out_fname, dpi=150, bbox_inches='tight')
            plt.close(fig)

        # -----------------------------------------------------------------
        # Summation across all hours
        # -----------------------------------------------------------------
        if plotted_hours:
            for h in plotted_hours:
                hour_idx = h
                with np.errstate(divide='ignore', invalid='ignore'):
                    final_norm = domain_pred_day[hour_idx] / domain_wt_day[hour_idx]
                final_norm[~np.isfinite(final_norm)] = 0.0

                mosaic_log = (final_norm * precip_std) + precip_mean
                mosaic_mm  = np.expm1(mosaic_log)

                # Only sum where ground truth isn't NaN
                mask_valid = ~np.isnan(domain_true_day[hour_idx])
                domain_pred_sum[mask_valid] += mosaic_mm[mask_valid]
                domain_true_sum[mask_valid] += domain_true_day[hour_idx][mask_valid]

    # ---------------------------------------------------------------------
    # 6) Plot total sum across entire date range
    # ---------------------------------------------------------------------
    total_max = max(domain_pred_sum.max(), domain_true_sum.max())
    if total_max <= 0:
        print("No valid data across the entire date range.")
        return

    fig, (ax_pred_sum, ax_true_sum) = plt.subplots(
        1, 2,
        figsize=(12, 6),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    for ax_ in (ax_pred_sum, ax_true_sum):
        ax_.set_extent(map_extent, crs=ccrs.PlateCarree())
        ax_.add_feature(cfeature.STATES, edgecolor='black', linewidth=1)
        ax_.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
        ax_.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)

    im_pred_sum = ax_pred_sum.imshow(
        domain_pred_sum,
        origin='lower',
        extent=map_extent,
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        vmin=0.0,
        vmax=total_max
    )
    ax_pred_sum.set_title(f"Total Model Output\n({_fmt_date(*start_day)} - {_fmt_date(*end_day)})")

    im_true_sum = ax_true_sum.imshow(
        domain_true_sum,
        origin='lower',
        extent=map_extent,
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        vmin=0.0,
        vmax=total_max
    )
    ax_true_sum.set_title(f"Total Ground Truth\n({_fmt_date(*start_day)} - {_fmt_date(*end_day)})")

    fig.subplots_adjust(bottom=0.15)
    cbar2 = fig.colorbar(im_true_sum, ax=(ax_pred_sum, ax_true_sum), orientation='horizontal', fraction=0.046, pad=0.08)
    cbar2.set_label("Total Precip (mm)")

    # Same naming as run_inference
    out_total = out_dir / f"inference_total_{_fmt_date(*start_day)}_{_fmt_date(*end_day)}.png"
    plt.savefig(out_total, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nAll inference plots saved to: {out_dir}")
    print("Done!")


if __name__ == "__main__":
    # Example usage:
    example_checkpoint = CHECKPOINTS_DIR / "best" / "best_model.pt"
    run_inference_manhattan(
        start_day=(2019, 1, 15),
        end_day=(2019, 1, 19),
        checkpoint_path=str(example_checkpoint)
    )
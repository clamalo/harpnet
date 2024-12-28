# File: /run_inference_manhattan.py

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
    SECONDARY_TILES,
    COARSE_RESOLUTION
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

    c1 = N // 2 - 1
    c2 = N // 2
    rows, cols = np.indices((N, N))

    dist1 = np.abs(rows - c1) + np.abs(cols - c1)
    dist2 = np.abs(rows - c1) + np.abs(cols - c2)
    dist3 = np.abs(rows - c2) + np.abs(cols - c1)
    dist4 = np.abs(rows - c2) + np.abs(cols - c2)

    dist = np.minimum(np.minimum(dist1, dist2), np.minimum(dist3, dist4))
    max_dist = dist.max()
    mask = 1.0 - (dist / max_dist) if max_dist > 0 else np.ones((N, N), dtype=np.float32)
    return mask.astype(np.float32)


def _date_range(sd=(1980, 9, 3), ed=(1980, 9, 5)):
    sy, sm, sd_ = sd
    ey, em, ed_ = ed
    start_dt = date(sy, sm, sd_)
    end_dt = date(ey, em, ed_)
    delta_ = (end_dt - start_dt).days
    for i in range(delta_ + 1):
        d_ = start_dt + timedelta(days=i)
        yield (d_.year, d_.month, d_.day)


def _upsample_coarse_with_crop_torch(coarse_tensor: torch.Tensor, final_shape: tuple) -> torch.Tensor:
    """
    Same approach as in run_inference.py: upsample to a bigger padded shape, 
    then center-crop to 'final_shape'.
    """
    _, _, cLat, cLon = coarse_tensor.shape
    fLat, fLon = final_shape

    ratio = int(round(COARSE_RESOLUTION / FINE_RESOLUTION))
    upsample_size_lat = cLat * ratio
    upsample_size_lon = cLon * ratio

    upsampled_t = F.interpolate(
        coarse_tensor,
        size=(upsample_size_lat, upsample_size_lon),
        mode='bilinear',
        align_corners=False
    )

    lat_diff = upsample_size_lat - fLat
    lon_diff = upsample_size_lon - fLon
    top = lat_diff // 2
    left = lon_diff // 2
    bottom = top + fLat
    right = left + fLon

    return upsampled_t[:, :, top:bottom, left:right]


def run_inference_manhattan(
    start_day=(1980, 9, 3),
    end_day=(1980, 9, 3),
    checkpoint_path=None
):
    """
    Run model inference for [start_day..end_day], using a staggered tiling approach 
    for primary + secondary tiles (Manhattan weighting). The ground truth is stored 
    unchanged.

    The main difference from run_inference.py is how we combine overlapping tiles 
    via the Manhattan weights, but we still apply the "upsample bigger + crop" logic 
    so the coarse input matches the fine resolution + domain exactly after cropping.
    """
    def _fmt_date(yr, mn, dy):
        return f"{yr:04d}{mn:02d}{dy:02d}"

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model_module = importlib.import_module(MODEL_NAME)
    ModelClass = model_module.Model

    general_ckpt_data = torch.load(checkpoint_path, map_location=DEVICE)
    general_ckpt_state_dict = general_ckpt_data.get('model_state_dict', None)
    if general_ckpt_state_dict is None:
        raise ValueError(f"Could not load 'model_state_dict' from {checkpoint_path}")

    tile_dict = get_tile_dict()
    tile_ids = sorted(tile_dict.keys())
    tile_models = {}

    for tid in tile_ids:
        tile_model = ModelClass().to(DEVICE)
        tile_specific_ckpt = CHECKPOINTS_DIR / "best" / f"{tid}_best.pt"

        if tile_specific_ckpt.exists():
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

    domain_pred_sum = np.zeros((nLat, nLon), dtype=np.float32)
    domain_true_sum = np.zeros((nLat, nLon), dtype=np.float32)

    ds_elev = xr.open_dataset(ELEVATION_FILE)
    if 'Y' in ds_elev.dims and 'X' in ds_elev.dims:
        ds_elev = ds_elev.rename({'Y': 'lat', 'X': 'lon'})

    tile_elevations = {}
    for tid in tile_ids:
        _, _, lat_fine, lon_fine = tile_coordinates(tid)
        elev_vals = ds_elev.topo.interp(lat=lat_fine, lon=lon_fine, method='nearest').values
        elev_vals = np.nan_to_num(elev_vals, nan=0.0).astype(np.float32)
        tile_elevations[tid] = elev_vals / 8848.9
    ds_elev.close()

    manhattan_mask = tile_weight_mask(TILE_SIZE)
    out_dir = FIGURES_DIR / "inference_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _date_range_inner(sd, ed):
        sy, sm, sd_ = sd
        ey, em, ed_ = ed
        start_dt = date(sy, sm, sd_)
        end_dt = date(ey, em, ed_)
        delta_ = (end_dt - start_dt).days
        for i_ in range(delta_ + 1):
            d_ = start_dt + timedelta(days=i_)
            yield (d_.year, d_.month, d_.day)

    for (year, month, day) in _date_range_inner(start_day, end_day):
        day_str = f"{year:04d}-{month:02d}-{day:02d}"
        day_tag = _fmt_date(year, month, day)

        domain_pred_day = np.zeros((24, nLat, nLon), dtype=np.float32)
        domain_wt_day = np.zeros((24, nLat, nLon), dtype=np.float32)
        domain_true_day = np.full((24, nLat, nLon), np.nan, dtype=np.float32)

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

            coarse_ds = ds_day.tp.interp(lat=lat_coarse, lon=lon_coarse, method="linear")
            fine_ds = ds_day.tp.interp(lat=lat_fine, lon=lon_fine, method="linear")

            c_vals = coarse_ds.values
            f_vals = fine_ds.values
            elev_tile = tile_elevations[tid]

            for i_hour, h_val in enumerate(ds_hours):
                hour_idx = h_val
                lat_indices = np.searchsorted(lat_global, lat_fine)
                lon_indices = np.searchsorted(lon_global, lon_fine)

                # 1) ground truth
                domain_true_day[hour_idx,
                                lat_indices[0]:lat_indices[-1]+1,
                                lon_indices[0]:lon_indices[-1]+1] = f_vals[i_hour]

                # 2) model inference or skip zeros
                coarse_precip_mm = c_vals[i_hour].astype(np.float32)
                if (not INCLUDE_ZEROS) and (coarse_precip_mm.max() < 0.1):
                    fill_val = -precip_mean / precip_std
                    domain_pred_day[hour_idx,
                                    lat_indices[0]:lat_indices[-1]+1,
                                    lon_indices[0]:lon_indices[-1]+1] += (fill_val * manhattan_mask)
                    domain_wt_day[hour_idx,
                                  lat_indices[0]:lat_indices[-1]+1,
                                  lon_indices[0]:lon_indices[-1]+1] += manhattan_mask
                    continue

                coarse_precip_log = np.log1p(coarse_precip_mm)
                coarse_precip_norm = (coarse_precip_log - precip_mean) / precip_std

                coarse_tensor = torch.from_numpy(coarse_precip_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
                upsampled_cropped_t = _upsample_coarse_with_crop_torch(
                    coarse_tensor,
                    final_shape=elev_tile.shape
                )
                elev_tile_t = torch.from_numpy(elev_tile).unsqueeze(0).unsqueeze(0).to(DEVICE)
                model_input = torch.cat([upsampled_cropped_t, elev_tile_t], dim=1)

                with torch.no_grad():
                    pred_t = tile_model(model_input)
                pred_arr_norm = pred_t.squeeze().cpu().numpy()

                # Weighted sum with Manhattan mask
                domain_pred_day[hour_idx,
                                lat_indices[0]:lat_indices[-1]+1,
                                lon_indices[0]:lon_indices[-1]+1] += (pred_arr_norm * manhattan_mask)
                domain_wt_day[hour_idx,
                              lat_indices[0]:lat_indices[-1]+1,
                              lon_indices[0]:lon_indices[-1]+1] += manhattan_mask

        ds.close()

        plotted_hours = sorted(hour_to_index.keys())
        for h_ in plotted_hours:
            hour_idx = h_

            # Merge the weighted sum
            with np.errstate(divide='ignore', invalid='ignore'):
                mosaic_norm = domain_pred_day[hour_idx] / domain_wt_day[hour_idx]
            mosaic_norm[~np.isfinite(mosaic_norm)] = 0.0

            # Un-normalize
            mosaic_log = (mosaic_norm * precip_std) + precip_mean
            mosaic_mm = np.expm1(mosaic_log)

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

            out_fname = out_dir / f"inference_{day_tag}_{hour_idx:02d}.png"
            plt.savefig(out_fname, dpi=150, bbox_inches='tight')
            plt.close(fig)

        # Summation
        if plotted_hours:
            for h_ in plotted_hours:
                hour_idx = h_
                with np.errstate(divide='ignore', invalid='ignore'):
                    final_norm = domain_pred_day[hour_idx] / domain_wt_day[hour_idx]
                final_norm[~np.isfinite(final_norm)] = 0.0

                mosaic_log = (final_norm * precip_std) + precip_mean
                mosaic_mm = np.expm1(mosaic_log)
                mask_valid = ~np.isnan(domain_true_day[hour_idx])
                domain_pred_sum[mask_valid] += mosaic_mm[mask_valid]
                domain_true_sum[mask_valid] += domain_true_day[hour_idx][mask_valid]

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

    out_total = out_dir / f"inference_total_{_fmt_date(*start_day)}_{_fmt_date(*end_day)}.png"
    plt.savefig(out_total, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nAll inference plots saved to: {out_dir}")
    print("Done!")


if __name__ == "__main__":
    # Example usage:
    example_checkpoint = CHECKPOINTS_DIR / "best" / "best_model.pt"
    run_inference_manhattan(
        start_day=(2019, 2, 8),
        end_day=(2019, 2, 23),
        checkpoint_path=str(example_checkpoint)
    )
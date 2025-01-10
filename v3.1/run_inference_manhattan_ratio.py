# File: run_inference_manhattan_gfs.py

import importlib
import torch
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch.nn.functional as F
import requests
import cfgrib

from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

from config import (
    ELEVATION_FILE,
    MODEL_NAME,
    DEVICE,
    FIGURES_DIR,
    PROCESSED_DIR,
    INCLUDE_ZEROS,
    CHECKPOINTS_DIR,
    FINE_RESOLUTION,
    TILE_SIZE,
    COARSE_RESOLUTION,
    PRECIP_THRESHOLD,
    SECONDARY_TILES
)
from tiles import (
    get_tile_dict,
    tile_coordinates
)

#############################################
# ECMWF Retrieval Helper Function
#############################################
from ecmwf.opendata import Client

def fetch_ecmwf_data(date_str: str, cycle: str, frames, ingest: bool = True) -> xr.Dataset:
    """
    Download ECMWF data using ecmwf.opendata, storing the output in 'data/ecmwf.grib'.
    Then open it with cfgrib and compute the precipitation for each forecast step
    by subtracting the cumulative total from the previous step.

    We avoid forcing a direct rename from 'step' to 'valid_time' if 'valid_time' 
    already exists. Instead, we detect whichever dimension is the "time" dimension. 
    Then we do:
      1) Sort by that dimension
      2) Convert 'tp' from cumulative to step precipitation

    Args:
        date_str (str): 'YYYYMMDD'
        cycle (str):    '00', '06', '12', '18'
        frames (list or int): Forecast steps, e.g. [0, 3, 6, 9, ...]
        ingest (bool):  Whether to actually retrieve the data or assume a local file is present.

    Returns:
        xr.Dataset: ECMWF dataset with 'tp' containing precipitation (mm) for each forecast step.
    """
    single_file_path = 'data/ecmwf.grib'
    os.makedirs('data', exist_ok=True)

    if ingest and os.path.exists(single_file_path):
        os.remove(single_file_path)

    if ingest:
        client = Client("ecmwf", beta=True)
        parameters = ['tp']

        if isinstance(frames, range):
            frames = list(frames)

        client.retrieve(
            date=date_str,
            time=cycle,
            step=frames,
            stream="oper",
            type="fc",
            levtype="sfc",
            param=parameters,
            target=single_file_path
        )

    ds = xr.open_dataset(single_file_path, engine='cfgrib')

    # Detect the time dimension
    possible_time_dims = ['valid_time', 'time', 'step']
    time_dim = None
    for dim in possible_time_dims:
        if dim in ds.dims:
            time_dim = dim
            break

    if not time_dim:
        raise ValueError("Could not find a recognizable time dimension in the ECMWF dataset!")

    # Sort by that time dimension
    ds = ds.sortby(time_dim)

    # Convert 'tp' to mm if needed, then turn it into step-accumulation
    ds['cum_tp'] = ds['tp'] * 1000.0
    tp_vals = ds['tp'].values.copy()
    cum_vals = ds['cum_tp'].values.copy()

    tp_vals[0] = cum_vals[0]
    for i in range(1, len(tp_vals)):
        tp_vals[i] = cum_vals[i] - cum_vals[i - 1]
    ds['tp'].values[:] = tp_vals

    # Rename lat/lon for consistency
    if 'latitude' in ds.coords and 'longitude' in ds.coords:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})

    # Sort lat ascending
    if 'lat' in ds.coords:
        ds = ds.sortby('lat', ascending=True)

    # Convert lon from [0,360) to [-180,180)
    if 'lon' in ds.coords and ds['lon'].max() > 180:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')

    return ds


#############################################
# GFS Retrieval Helper Functions
#############################################
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


def fetch_gfs_data(date_str: str, cycle: str, fhr: int, out_file: Path) -> xr.Dataset:
    """
    Download a single GFS GRIB2 file containing APCP (accumulated precip) from NOMADS
    and load it as an xarray Dataset using cfgrib.
    
    Args:
        date_str (str):  'YYYYMMDD', e.g. '20250105'
        cycle (str):     '00', '06', '12', '18'
        fhr (int):       Forecast hour, e.g. 3, 6, 9, ...
        out_file (Path): Local path to save the downloaded GRIB2 file.

    Returns:
        xr.Dataset: The loaded dataset with APCP_surface in mm.
    """
    base_url = (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        "?dir=%2Fgfs.{DATE}%2F{CYCLE}%2Fatmos"
        "&file=gfs.t{CYCLE}z.pgrb2.0p25.f{FHR:03d}"
        "&var_APCP=on&lev_surface=on"
    )
    url = base_url.format(DATE=date_str, CYCLE=cycle, FHR=fhr)

    if not out_file.parent.exists():
        out_file.parent.mkdir(parents=True, exist_ok=True)

    if not out_file.exists():
        print(f"Fetching GFS from {url}")
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            r.raise_for_status()
        with open(out_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {out_file}")
    else:
        print(f"Found existing file: {out_file}")

    ds = xr.open_dataset(
        out_file,
        engine="cfgrib",
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}}
    )
    # Convert longitudes from [0,360) to [-180,180)
    if 'longitude' in ds.coords:
        if ds['longitude'].max() > 180:
            ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
            ds = ds.sortby('longitude')

    return ds


def compute_hourly_precip(ds_current: xr.Dataset, ds_prev: xr.Dataset, fhr: int) -> xr.DataArray:
    """
    GFS's APCP_surface is typically accumulation from the start of the forecast.
    This function returns either:
      - ds_current['tp'] directly if fhr is not a multiple of 6 (or ds_prev is None).
      - ds_current['tp'] - ds_prev['tp'] if fhr is a multiple of 6.

    We clip negative differences to 0.0 just in case of minor numerical issues.

    Args:
        ds_current (xr.Dataset): The current forecast hour dataset.
        ds_prev (xr.Dataset):    The previous forecast hour dataset, or None if not available.
        fhr (int):               The forecast hour, e.g. 0, 3, 6, 9, etc.

    Returns:
        xr.DataArray: The instantaneous precipitation (mm) for this forecast hour.
    """
    if ds_prev is None or fhr % 6 != 0:
        return ds_current['tp']
    else:
        diff = ds_current['tp'] - ds_prev['tp']
        diff = diff.where(diff >= 0.0, 0.0)  # clip negative differences
        return diff


#############################################
# Main Inference Function
#############################################
def run_inference_manhattan_gfs(
    date_str: str = "20250105",
    cycle: str = "12",
    max_fhr: int = 24,   # e.g. go from f003 up to f024
    checkpoint_path=None,
    data_source: str = "GFS",
    ecmwf_ingest: bool = True
):
    """
    Similar to run_inference_manhattan.py, but now with an option to use live GFS or ECMWF data 
    for precipitation input.

    For GFS:
      1) For a given date_str (YYYYMMDD) and cycle (00/06/12/18), we iterate from f000 up to f{max_fhr} in steps of 3.
      2) For each forecast hour, we download the GFS file, parse the APCP_surface variable 
         (accumulated precipitation).
      3) Compute the precipitation for each time step by either:
         - using the direct accumulation if the forecast hour is not a multiple of 6,
         - or subtracting the previous accumulation (if the forecast hour is multiple of 6).
      4) Interpolate that coarse GFS array onto each tile, then combine the tile-based 
         downscaling predictions via Manhattan weighting.
      5) Output an hourly map plus a total accumulation map across the entire forecast window.

    For ECMWF:
      1) We retrieve the data via the ecmwf.opendata Client, which downloads a single GRIB file
         with all forecast steps included.
      2) We convert 'tp' from cumulative to step precipitation.
      3) Then for each forecast step, we do the same tile-based Manhattan weighting approach.

    Additionally, for each time frame, we compute the "downscale ratio":
      ratio = (high-resolution bilinear interpolation) / (downscaled model)
    This is plotted alongside the model output and the interpolated truth, with ratio clipped 
    to [0.5, 2] for visualization.

    Args:
        date_str (str):       'YYYYMMDD', e.g. '20250105'
        cycle (str):          '00', '06', '12', or '18'
        max_fhr (int):        Maximum forecast hour to process (in steps of 3) for GFS.
                              For ECMWF, this indicates the maximum step to request.
        checkpoint_path (str or Path): Path to the model checkpoint to load.
        data_source (str):    "GFS" (default) or "ECMWF"
        ecmwf_ingest (bool):  Whether to retrieve ECMWF data from the server or assume local file 
                              (ignored if data_source="GFS").
    """
    def _fmt_label(d_str, cyc, f_hr):
        return f"{d_str} {cyc}z f{f_hr:03d}"

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load the model
    model_module = importlib.import_module(MODEL_NAME)
    ModelClass = model_module.Model

    general_ckpt_data = torch.load(checkpoint_path, map_location=DEVICE)
    general_ckpt_state_dict = general_ckpt_data.get('model_state_dict', None)
    if general_ckpt_state_dict is None:
        raise ValueError(f"Could not load 'model_state_dict' from {checkpoint_path}")

    # Load tile-specific fine-tuned weights if available
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

    # Load normalization stats
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

    # Determine domain bounding box
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

    # Prepare arrays for storing total accumulation
    domain_pred_sum = np.zeros((nLat, nLon), dtype=np.float32)
    domain_true_sum = np.zeros((nLat, nLon), dtype=np.float32)

    # Load elevation data
    ds_elev = xr.open_dataset(ELEVATION_FILE)
    if 'Y' in ds_elev.dims and 'X' in ds_elev.dims:
        ds_elev = ds_elev.rename({'Y': 'lat', 'X': 'lon'})
    tile_elevations = {}
    for tid in tile_ids:
        # tile_dict[tid] = (min_lat, max_lat, min_lon, max_lon, whatever)
        # tile_coordinates(tid) should return the 4 needed arrays:
        #    lat_coarse, lon_coarse, lat_fine, lon_fine
        lat_coarse, lon_coarse, lat_fine, lon_fine = tile_coordinates(tid)
        elev_vals = ds_elev.topo.interp(lat=lat_fine, lon=lon_fine, method='nearest').values
        elev_vals = np.nan_to_num(elev_vals, nan=0.0).astype(np.float32)
        tile_elevations[tid] = elev_vals / 8848.9
    ds_elev.close()

    manhattan_mask = tile_weight_mask(TILE_SIZE)

    # Output directory
    out_dir = FIGURES_DIR / "inference_output_gfs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ##################################################
    # ECMWF Option
    ##################################################
    if data_source.upper() == "ECMWF":
        # We'll treat frames as [0, 3, 6, ..., max_fhr]
        frames = list(range(0, max_fhr + 1, 3))
        ds_ecmwf = fetch_ecmwf_data(date_str, cycle, frames, ecmwf_ingest)

        # Detect which dimension is the time dimension
        possible_time_dims = ['valid_time', 'time', 'step']
        ecmwf_time_dim = None
        for dim in possible_time_dims:
            if dim in ds_ecmwf.dims:
                ecmwf_time_dim = dim
                break
        if not ecmwf_time_dim:
            raise ValueError("ECMWF dataset: no recognizable time dimension found after fetch.")

        n_times = ds_ecmwf.sizes[ecmwf_time_dim]

        for i in range(n_times):
            # Construct a label for plotting
            hour_label = f"{date_str} {cycle}z T{i:02d}"
            precip_da = ds_ecmwf['tp'].isel({ecmwf_time_dim: i})

            # Prepare arrays for this hour
            domain_pred_hour = np.zeros((nLat, nLon), dtype=np.float32)
            domain_wt_hour = np.zeros((nLat, nLon), dtype=np.float32)
            domain_true_hour = np.full((nLat, nLon), np.nan, dtype=np.float32)

            for tid in tqdm(tile_ids, desc=f"ECMWF step {i:02d}"):
                # For each tile, retrieve lat/lon arrays
                lat_coarse, lon_coarse, lat_fine, lon_fine = tile_coordinates(tid)
                tile_model = tile_models[tid]
                elev_tile = tile_elevations[tid]

                # Interpolate onto coarse grid
                ds_tile_coarse = precip_da.interp(lat=lat_coarse, lon=lon_coarse, method="linear")
                c_vals = ds_tile_coarse.values.astype(np.float32)

                # Interpolate for "truth" on fine grid
                ds_tile_fine = precip_da.interp(lat=lat_fine, lon=lon_fine, method="linear")
                f_vals = ds_tile_fine.values.astype(np.float32)

                # Insert the "truth" into the domain array
                lat_indices = np.searchsorted(lat_global, lat_fine)
                lon_indices = np.searchsorted(lon_global, lon_fine)
                domain_true_hour[lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1] = f_vals

                # Possibly skip zeros
                if (not INCLUDE_ZEROS) and (c_vals.max() < PRECIP_THRESHOLD):
                    fill_val_norm = -precip_mean / precip_std
                    domain_pred_hour[lat_indices[0]:lat_indices[-1]+1,
                                     lon_indices[0]:lon_indices[-1]+1] += (fill_val_norm * manhattan_mask)
                    domain_wt_hour[lat_indices[0]:lat_indices[-1]+1,
                                   lon_indices[0]:lon_indices[-1]+1] += manhattan_mask
                    continue

                # Normal pipeline
                c_log = np.log1p(c_vals)
                c_norm = (c_log - precip_mean) / precip_std
                coarse_tensor = torch.from_numpy(c_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
                upsampled_cropped_t = _upsample_coarse_with_crop_torch(
                    coarse_tensor,
                    final_shape=elev_tile.shape
                )

                elev_tile_t = torch.from_numpy(elev_tile).unsqueeze(0).unsqueeze(0).to(DEVICE)
                model_input = torch.cat([upsampled_cropped_t, elev_tile_t], dim=1)
                with torch.no_grad():
                    pred_t = tile_model(model_input)
                pred_arr_norm = pred_t.squeeze().cpu().numpy()

                # Weighted sum
                domain_pred_hour[lat_indices[0]:lat_indices[-1]+1,
                                 lon_indices[0]:lon_indices[-1]+1] += (pred_arr_norm * manhattan_mask)
                domain_wt_hour[lat_indices[0]:lat_indices[-1]+1,
                               lon_indices[0]:lon_indices[-1]+1] += manhattan_mask

            # Combine
            with np.errstate(divide='ignore', invalid='ignore'):
                final_norm = domain_pred_hour / domain_wt_hour
            final_norm[~np.isfinite(final_norm)] = 0.0

            # Un-normalize
            mosaic_log = (final_norm * precip_std) + precip_mean
            mosaic_mm = np.expm1(mosaic_log)

            # Compute downscale ratio: (bilinear interp) / (downscaled model)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = mosaic_mm / domain_true_hour
            ratio[~np.isfinite(ratio)] = np.nan

            # Plot all three: model, "truth" (interpolated), ratio
            fig, (ax_pred, ax_true, ax_ratio) = plt.subplots(
                1, 3,
                figsize=(18, 6),
                subplot_kw={'projection': ccrs.PlateCarree()}
            )
            for ax_ in (ax_pred, ax_true, ax_ratio):
                ax_.set_extent(map_extent, crs=ccrs.PlateCarree())
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
            ax_pred.set_title(f"Downscaled Model\n{hour_label}")
            cbar_pred = fig.colorbar(
                im_pred, ax=ax_pred, orientation='horizontal', fraction=0.046, pad=0.07
            )
            cbar_pred.set_label("Precip (mm)")

            im_true = ax_true.imshow(
                domain_true_hour,
                origin='lower',
                extent=map_extent,
                transform=ccrs.PlateCarree(),
                cmap='viridis',
                vmin=0.0,
                vmax=10.0
            )
            ax_true.set_title(f"ECMWF (Interpolated)\n{hour_label}")
            cbar_true = fig.colorbar(
                im_true, ax=ax_true, orientation='horizontal', fraction=0.046, pad=0.07
            )
            cbar_true.set_label("Precip (mm)")

            im_ratio = ax_ratio.imshow(
                ratio,
                origin='lower',
                extent=map_extent,
                transform=ccrs.PlateCarree(),
                cmap='RdBu_r',
                vmin=0,
                vmax=2.0
            )
            ax_ratio.set_title(f"Downscale Ratio\n(Model / ECMWF)")
            cbar_ratio = fig.colorbar(
                im_ratio, ax=ax_ratio, orientation='horizontal', fraction=0.046, pad=0.07
            )
            cbar_ratio.set_label("Ratio")

            fig.subplots_adjust(wspace=0.2, bottom=0.1)
            out_dir_fname = out_dir / f"inference_ECMWF_{date_str}_{cycle}_t{i:02d}.png"
            plt.savefig(out_dir_fname, dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Accumulate total
            mask_valid = ~np.isnan(domain_true_hour)
            domain_pred_sum[mask_valid] += mosaic_mm[mask_valid]
            domain_true_sum[mask_valid] += domain_true_hour[mask_valid]

    ##################################################
    # GFS Option
    ##################################################
    else:
        ds_prev = None
        forecast_hours = list(range(0, max_fhr + 1, 3))
        if 0 not in forecast_hours:
            forecast_hours.insert(0, 0)
        forecast_hours = sorted(set(forecast_hours))

        for fhr in forecast_hours:
            out_file = PROCESSED_DIR / "gfs_grib" / f"{date_str}_{cycle}_f{fhr:03d}.grib2"
            try:
                ds_current = fetch_gfs_data(date_str, cycle, fhr, out_file)
            except Exception as e:
                print(f"Failed to retrieve or parse GFS data for f{fhr:03d}: {e}")
                ds_prev = None
                continue

            # Convert from accumulated to instantaneous precipitation
            precip_da = compute_hourly_precip(ds_current, ds_prev, fhr)
            ds_prev = ds_current

            if precip_da.sizes.get('time', 1) > 1:
                precip_da = precip_da.isel(time=0)

            lat_name = 'latitude' if 'latitude' in precip_da.coords else 'lat'
            lon_name = 'longitude' if 'longitude' in precip_da.coords else 'lon'
            precip_da = precip_da.rename({lat_name: 'lat', lon_name: 'lon'})

            domain_pred_hour = np.zeros((nLat, nLon), dtype=np.float32)
            domain_wt_hour = np.zeros((nLat, nLon), dtype=np.float32)
            domain_true_hour = np.full((nLat, nLon), np.nan, dtype=np.float32)

            hour_label = _fmt_label(date_str, cycle, fhr)
            print(f"Running downscaling for GFS {hour_label} ...")

            for tid in tqdm(tile_ids, desc=f"GFS f{fhr:03d}"):
                lat_coarse, lon_coarse, lat_fine, lon_fine = tile_coordinates(tid)
                tile_model = tile_models[tid]
                elev_tile = tile_elevations[tid]

                ds_tile_coarse = precip_da.interp(lat=lat_coarse, lon=lon_coarse, method="linear")
                c_vals = ds_tile_coarse.values.astype(np.float32)

                ds_tile_fine = precip_da.interp(lat=lat_fine, lon=lon_fine, method="linear")
                f_vals = ds_tile_fine.values.astype(np.float32)

                lat_indices = np.searchsorted(lat_global, lat_fine)
                lon_indices = np.searchsorted(lon_global, lon_fine)
                domain_true_hour[lat_indices[0]:lat_indices[-1]+1, lon_indices[0]:lon_indices[-1]+1] = f_vals

                if (not INCLUDE_ZEROS) and (c_vals.max() < PRECIP_THRESHOLD):
                    fill_val_norm = -precip_mean / precip_std
                    domain_pred_hour[lat_indices[0]:lat_indices[-1]+1,
                                     lon_indices[0]:lon_indices[-1]+1] += (fill_val_norm * manhattan_mask)
                    domain_wt_hour[lat_indices[0]:lat_indices[-1]+1,
                                   lon_indices[0]:lon_indices[-1]+1] += manhattan_mask
                    continue

                c_log = np.log1p(c_vals)
                c_norm = (c_log - precip_mean) / precip_std
                coarse_tensor = torch.from_numpy(c_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
                upsampled_cropped_t = _upsample_coarse_with_crop_torch(
                    coarse_tensor,
                    final_shape=elev_tile.shape
                )

                elev_tile_t = torch.from_numpy(elev_tile).unsqueeze(0).unsqueeze(0).to(DEVICE)
                model_input = torch.cat([upsampled_cropped_t, elev_tile_t], dim=1)
                with torch.no_grad():
                    pred_t = tile_model(model_input)
                pred_arr_norm = pred_t.squeeze().cpu().numpy()

                domain_pred_hour[lat_indices[0]:lat_indices[-1]+1,
                                 lon_indices[0]:lon_indices[-1]+1] += (pred_arr_norm * manhattan_mask)
                domain_wt_hour[lat_indices[0]:lat_indices[-1]+1,
                               lon_indices[0]:lon_indices[-1]+1] += manhattan_mask

            with np.errstate(divide='ignore', invalid='ignore'):
                final_norm = domain_pred_hour / domain_wt_hour
            final_norm[~np.isfinite(final_norm)] = 0.0

            # Un-normalize
            mosaic_log = (final_norm * precip_std) + precip_mean
            mosaic_mm = np.expm1(mosaic_log)

            # Compute downscale ratio: (bilinear interp) / (downscaled model)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = mosaic_mm / domain_true_hour
            ratio[~np.isfinite(ratio)] = np.nan

            # Plot
            fig, (ax_pred, ax_true, ax_ratio) = plt.subplots(
                1, 3,
                figsize=(18, 6),
                subplot_kw={'projection': ccrs.PlateCarree()}
            )
            for ax_ in (ax_pred, ax_true, ax_ratio):
                ax_.set_extent(map_extent, crs=ccrs.PlateCarree())
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
            ax_pred.set_title(f"Downscaled Model\n{hour_label}")
            cbar_pred = fig.colorbar(
                im_pred, ax=ax_pred, orientation='horizontal', fraction=0.046, pad=0.07
            )
            cbar_pred.set_label("Precip (mm)")

            im_true = ax_true.imshow(
                domain_true_hour,
                origin='lower',
                extent=map_extent,
                transform=ccrs.PlateCarree(),
                cmap='viridis',
                vmin=0.0,
                vmax=10.0
            )
            ax_true.set_title(f"GFS (Interpolated)\n{hour_label}")
            cbar_true = fig.colorbar(
                im_true, ax=ax_true, orientation='horizontal', fraction=0.046, pad=0.07
            )
            cbar_true.set_label("Precip (mm)")

            im_ratio = ax_ratio.imshow(
                ratio,
                origin='lower',
                extent=map_extent,
                transform=ccrs.PlateCarree(),
                cmap='RdBu_r',
                vmin=0,
                vmax=2.0
            )
            ax_ratio.set_title("Downscale Ratio\n(Model / GFS)")
            cbar_ratio = fig.colorbar(
                im_ratio, ax=ax_ratio, orientation='horizontal', fraction=0.046, pad=0.07
            )
            cbar_ratio.set_label("Ratio")

            fig.subplots_adjust(wspace=0.2, bottom=0.1)
            out_dir_fname = out_dir / f"inference_{date_str}_{cycle}_f{fhr:03d}.png"
            plt.savefig(out_dir_fname, dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Accumulate total
            mask_valid = ~np.isnan(domain_true_hour)
            domain_pred_sum[mask_valid] += mosaic_mm[mask_valid]
            domain_true_sum[mask_valid] += domain_true_hour[mask_valid]

    ##################################################
    #  Final: plot the total accumulation
    ##################################################
    total_max = max(domain_pred_sum.max(), domain_true_sum.max())
    if total_max <= 0:
        print("No valid data across the forecast range.")
        return

    fig, (ax_pred_sum, ax_true_sum) = plt.subplots(
        1, 2,
        figsize=(12, 6),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    for ax_ in (ax_pred_sum, ax_true_sum):
        ax_.set_extent(map_extent, crs=ccrs.PlateCarree())
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
    ax_pred_sum.set_title(f"Total Model Output\n{date_str} {cycle}z")

    im_true_sum = ax_true_sum.imshow(
        domain_true_sum,
        origin='lower',
        extent=map_extent,
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        vmin=0.0,
        vmax=total_max
    )
    ax_true_sum.set_title(f"Total {data_source} Accumulation\n{date_str} {cycle}z")

    fig.subplots_adjust(bottom=0.15)
    cbar2 = fig.colorbar(im_true_sum, ax=(ax_pred_sum, ax_true_sum), orientation='horizontal', fraction=0.046, pad=0.08)
    cbar2.set_label("Total Precip (mm)")

    out_total = out_dir / f"inference_total_{data_source}_{date_str}_{cycle}.png"
    plt.savefig(out_total, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nAll {data_source}-based inference plots saved to: {out_dir}")
    print("Done!")


if __name__ == "__main__":
    """
    Example usage:
      python run_inference_manhattan_gfs.py
        This will default to GFS with date_str='20250105', cycle='12', max_fhr=24,
        reading from CHECKPOINTS_DIR / 'best' / 'best_model.pt'.

      For ECMWF, something like:
        python run_inference_manhattan_gfs.py data_source=ECMWF ecmwf_ingest=True
    """
    example_checkpoint = CHECKPOINTS_DIR / "best" / "best_model.pt"
    # By default, we do GFS:
    run_inference_manhattan_gfs(
        date_str="20250107",
        cycle="00",
        max_fhr=120,
        checkpoint_path=str(example_checkpoint),
        data_source="GFS"
    )
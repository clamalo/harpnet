# File: /inference.py

import os
import math
import requests
import importlib
import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import USCOUNTIES

from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

# ECMWF open data client
try:
    from ecmwf.opendata import Client as ECMWFClient
except ImportError:
    ECMWFClient = None  # We'll handle this gracefully if data_source=ECMWF

from config import (
    RAW_DIR, ELEVATION_FILE, MODEL_NAME, DEVICE, FIGURES_DIR,
    PROCESSED_DIR, INCLUDE_ZEROS, CHECKPOINTS_DIR,
    FINE_RESOLUTION, TILE_SIZE, COARSE_RESOLUTION,
    PRECIP_THRESHOLD, SECONDARY_TILES
)
from tiles import (
    get_tile_dict,
    tile_coordinates
)


def _tile_weight_mask(N=TILE_SIZE):
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
    Upsample a coarse-resolution tensor to a larger padded size, then
    center-crop so that the final shape matches 'final_shape'.

    coarse_tensor: shape (1, 1, cLat, cLon)
    final_shape:   (fLat, fLon)
    Returns a torch.Tensor of shape (1, 1, fLat, fLon)
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


def _weatherbell_precip_colormap():
    """
    Creates a ListedColormap based on a WeatherBell-like precipitation palette.
    Used for GFS or ECMWF plotting if desired.
    """
    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap([
        '#ffffff', '#bdbfbd', '#aba5a5', '#838383', '#6e6e6e', '#b5fbab', '#95f58b', '#78f572',
        '#50f150', '#1eb51e', '#0ea10e', '#1464d3', '#2883f1', '#50a5f5', '#97d3fb', '#b5f1fb',
        '#fffbab', '#ffe978', '#ffc13c', '#ffa100', '#ff6000', '#ff3200', '#e11400', '#c10000',
        '#a50000', '#870000', '#643c31', '#8d6558', '#b58d83', '#c7a095', '#f1ddd3', '#cecbdc'
    ])
    cmap.set_over(color='#aca0c7')
    return cmap


def _fetch_ecmwf_data_frame(date_str: str, cycle_str: str, fhr: int, out_file: Path):
    """
    Download a single ECMWF GRIB file for the specified forecast hour from ecmwf.opendata,
    and load it as an xarray Dataset using cfgrib.

    The result is an ECMWF cumulative total precipitation from the start of the forecast
    up to 'fhr'.
    """
    if ECMWFClient is None:
        raise ImportError("ecmwf.opendata not installed or not importable. Cannot fetch ECMWF data.")

    client = ECMWFClient("ecmwf", beta=True)
    params = ['tp']  # total precipitation

    client.retrieve(
        date=date_str,       # e.g. "20250111"
        time=cycle_str,      # e.g. "12"
        step=[fhr],          # e.g. [3]
        stream="oper",
        type="fc",
        levtype="sfc",
        param=params,
        target=str(out_file) # e.g. "ecmwf_grib/ecmwf_20250111_12_f003.grib"
    )

    ds = xr.open_dataset(out_file, engine='cfgrib')
    # Convert lat/lon to [-180, 180), ascending lat
    if 'latitude' in ds.coords and 'longitude' in ds.coords:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    if 'lat' in ds.coords:
        ds = ds.sortby('lat', ascending=True)
    if 'lon' in ds.coords and ds['lon'].max() > 180:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')

    if 'tp' not in ds.variables:
        for var_ in ds.variables:
            if var_.lower() not in ('lat', 'lon'):
                ds = ds.rename({var_: 'tp'})
                break

    ds['tp'] = ds['tp'] * 1000.0  # ensure mm if originally in m
    return ds


def _compute_hourly_precip_ecmwf(ds_current: xr.Dataset, ds_prev: xr.Dataset, fhr: int) -> xr.DataArray:
    """
    ECMWF 'tp' is cumulative from the start of the forecast. For the first processed frame,
    there's no ds_prev, so we just return ds_current. If ds_prev is provided, we do ds_current - ds_prev.
    Negative diffs clipped to 0.0 just in case.
    """
    if ds_prev is None or fhr == 3:
        return ds_current['tp']
    else:
        diff = ds_current['tp'] - ds_prev['tp']
        diff = diff.where(diff >= 0.0, 0.0)
        return diff


def _fetch_gfs_data(date_str, cycle_str, fhr, out_file: Path):
    """
    Download a single GFS GRIB2 file containing APCP (accumulated precip) from NOMADS,
    storing it at out_file, and load via cfgrib.
    """
    base_url = (
        "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        "?dir=%2Fgfs.{DATE}%2F{CYCLE}%2Fatmos"
        "&file=gfs.t{CYCLE}z.pgrb2.0p25.f{FHR:03d}"
        "&var_APCP=on&lev_surface=on"
    )
    url = base_url.format(DATE=date_str, CYCLE=cycle_str, FHR=fhr)

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
    if 'longitude' in ds.coords:
        if ds['longitude'].max() > 180:
            ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby('longitude')
    return ds


def _compute_hourly_precip_gfs(ds_current: xr.Dataset, ds_prev: xr.Dataset, fhr: int) -> xr.DataArray:
    """
    GFS APCP_surface is accumulation from the start of the forecast.
    We return ds_current['tp'] if ds_prev is None or fhr not multiple of 6,
    else (ds_current['tp'] - ds_prev['tp']), negative diffs clipped to 0.0.
    """
    if ds_prev is None or fhr % 6 != 0:
        return ds_current['tp']
    else:
        diff = ds_current['tp'] - ds_prev['tp']
        diff = diff.where(diff >= 0.0, 0.0)
        return diff


def _parse_init_datestr(init_datestr: str) -> datetime:
    """
    Parse a string 'YYYYMMDDHH' into a datetime object.
    """
    if len(init_datestr) != 10:
        raise ValueError(f"init_datestr must be YYYYMMDDHH (10 chars). Got: {init_datestr}")
    year = int(init_datestr[0:4])
    month = int(init_datestr[4:6])
    day = int(init_datestr[6:8])
    hour = int(init_datestr[8:10])
    return datetime(year, month, day, hour)


def run_inference(
    data_source: str,
    init_datestr: str,
    hours: int,
    ratio: bool = False,
    colormap: str = "weatherbell"
):
    """
    Consolidated inference function for local NetCDF data, ECMWF realtime data, or GFS realtime data.
    Both ECMWF and GFS ingest one forecast frame at a time, saving GRIB to PROCESSED_DIR. Filenames
    do NOT change if ratio is toggled.

    Args:
        data_source (str): One of ["local", "ECMWF", "GFS"].
        init_datestr (str): "YYYYMMDDHH" string representing the initial time.
                            e.g. "2025011112" means 2025-01-11 12Z.
        hours (int): How many hours to process. The script plots frames in multiples of 3 up to 'hours'.
        ratio (bool): If True, plot a 3-panel figure (model, coarse/bilinear, ratio).
                      If False, plot a 2-panel figure (model, coarse/bilinear).
        colormap (str): Either "viridis" or "weatherbell". If "viridis", each 3-hourly plot
                        is scaled to [0, 5]. The accumulated total plot is scaled to [0, max_value].
                        If "weatherbell", uses the existing WeatherBell color palette and boundaries.
    """
    # --------------------------------------------
    # 1) Parse init date/time, create forecast steps
    # --------------------------------------------
    init_dt = _parse_init_datestr(init_datestr)  # e.g. 2025-01-11 12:00
    fcst_hours = list(range(3, hours + 1, 3))

    # --------------------------------------------
    # 2) Load model & tile-specific weights
    # --------------------------------------------
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    model_module = importlib.import_module(MODEL_NAME)
    ModelClass = model_module.Model

    gen_ckpt_path = CHECKPOINTS_DIR / "best" / "best_model.pt"
    if not gen_ckpt_path.exists():
        raise FileNotFoundError(f"General best_model.pt not found at {gen_ckpt_path}")
    general_ckpt_data = torch.load(gen_ckpt_path, map_location=device)
    general_state_dict = general_ckpt_data.get("model_state_dict", None)
    if general_state_dict is None:
        raise ValueError(f"Could not load 'model_state_dict' from {gen_ckpt_path}")

    tile_dict = get_tile_dict()
    tile_ids = sorted(tile_dict.keys())
    tile_models = {}
    for tid in tile_ids:
        tile_model = ModelClass().to(device)
        tile_ckpt = CHECKPOINTS_DIR / "best" / f"{tid}_best.pt"
        if tile_ckpt.exists():
            cdata = torch.load(tile_ckpt, map_location=device)
            st_dict = cdata.get("model_state_dict", None)
            if st_dict is not None:
                tile_model.load_state_dict(st_dict, strict=True)
            else:
                tile_model.load_state_dict(general_state_dict, strict=True)
        else:
            tile_model.load_state_dict(general_state_dict, strict=True)
        tile_model.eval()
        tile_models[tid] = tile_model

    # --------------------------------------------
    # 3) Load normalization stats
    # --------------------------------------------
    data_path = PROCESSED_DIR / "combined_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find {data_path}. Run data_preprocessing.py first.")
    with np.load(data_path) as d:
        precip_mean = d["precip_mean"].item()
        precip_std = d["precip_std"].item()

    print(f"Normalization stats (log space): mean={precip_mean:.4f}, std={precip_std:.4f}")

    # --------------------------------------------
    # 4) Prepare domain bounding box
    # --------------------------------------------
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
    nLat, nLon = len(lat_global), len(lon_global)
    map_extent = [min_lon_domain, max_lon_domain, min_lat_domain, max_lat_domain]

    # --------------------------------------------
    # 5) Load tile-based elevation data
    # --------------------------------------------
    ds_elev = xr.open_dataset(ELEVATION_FILE)
    if 'Y' in ds_elev.dims and 'X' in ds_elev.dims:
        ds_elev = ds_elev.rename({'Y': 'lat', 'X': 'lon'})
    tile_elevations = {}
    for tid in tile_ids:
        lat_coarse, lon_coarse, lat_fine, lon_fine = tile_coordinates(tid)
        elev_vals = ds_elev.topo.interp(lat=lat_fine, lon=lon_fine, method='nearest').values
        elev_vals = np.nan_to_num(elev_vals, nan=0.0).astype(np.float32)
        tile_elevations[tid] = elev_vals / 8848.9
    ds_elev.close()

    # --------------------------------------------
    # 6) We'll do frame-by-frame ingestion for GFS or ECMWF
    # --------------------------------------------
    ds_gfs_prev = None
    ds_ecmwf_prev = None

    out_dir = FIGURES_DIR / "inference_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------
    # Handle colormap logic
    # --------------------------------------------
    import matplotlib.colors as mcolors

    if colormap.lower() == "weatherbell":
        # Use the existing WeatherBell color palette and boundary norms
        wbell_cmap = _weatherbell_precip_colormap()
        bounds = [
            0.0, 0.25, 0.76, 1.27, 1.91, 2.54, 3.81, 5.08, 6.35, 7.62,
            10.16, 12.7, 15.24, 17.78, 20.32, 22.86, 25.4, 30.48, 35.56,
            40.64, 45.72, 50.8, 63.5, 76.2, 88.9, 101.6, 127.0, 152.4,
            177.8, 203.2, 228.6, 254.0
        ]
        wb_norm = mcolors.BoundaryNorm(bounds, ncolors=len(wbell_cmap.colors))
        cmap_hourly = wbell_cmap
        norm_hourly = wb_norm
        vmin_hourly = None
        vmax_hourly = None

        # For total plots, use same colormap & norms
        cmap_total = wbell_cmap
        norm_total = wb_norm
        # We'll still need 'bounds' for colorbar ticks.

    elif colormap.lower() == "viridis":
        # For 3-hourly frames, fixed vmin=0, vmax=5
        cmap_hourly = "viridis"
        norm_hourly = None
        vmin_hourly = 0
        vmax_hourly = 5

        # For total plots, we must determine vmax after we compute the total
        # We'll set these placeholders now and update them later:
        cmap_total = "viridis"
        norm_total = None
        # We won't define the total vmin/vmax yet; we'll do it once we have the max values.

        bounds = None  # We won't use the WeatherBell boundaries in viridis mode.
    else:
        raise ValueError("colormap must be either 'viridis' or 'weatherbell'.")

    # Accumulated totals
    domain_model_accum = np.zeros((nLat, nLon), dtype=np.float32)
    domain_coarse_accum = np.zeros((nLat, nLon), dtype=np.float32)

    # --------------------------------------------
    # 7) Iterate over forecast hours
    # --------------------------------------------
    for fhr in fcst_hours:
        valid_dt = init_dt + timedelta(hours=fhr)
        hr_label = valid_dt.strftime("%Y-%m-%d %H:%M UTC")

        # 7a) Retrieve one-frame cumulative precipitation
        if data_source.lower() == "ecmwf":
            date_str = init_dt.strftime("%Y%m%d")
            cycle_str = f"{init_dt.hour:02d}"
            ecmwf_dir = PROCESSED_DIR / "ecmwf_grib"
            ecmwf_dir.mkdir(parents=True, exist_ok=True)
            out_file = ecmwf_dir / f"ecmwf_{date_str}_{cycle_str}_f{fhr:03d}.grib"

            try:
                ds_current = _fetch_ecmwf_data_frame(date_str, cycle_str, fhr, out_file)
            except Exception as e:
                print(f"Failed to retrieve/parse ECMWF for f{fhr:03d}: {e}")
                ds_ecmwf_prev = None
                continue
            da_precip = _compute_hourly_precip_ecmwf(ds_current, ds_ecmwf_prev, fhr)
            ds_ecmwf_prev = ds_current

        elif data_source.lower() == "gfs":
            date_str = init_dt.strftime("%Y%m%d")
            cycle_str = f"{init_dt.hour:02d}"
            gfs_dir = PROCESSED_DIR / "gfs_grib"
            gfs_dir.mkdir(parents=True, exist_ok=True)
            out_file = gfs_dir / f"{date_str}_{cycle_str}_f{fhr:03d}.grib2"

            try:
                ds_current = _fetch_gfs_data(date_str, cycle_str, fhr, out_file)
            except Exception as e:
                print(f"Failed to retrieve/parse GFS for f{fhr:03d}: {e}")
                ds_gfs_prev = None
                continue
            if 'APCP_surface' in ds_current.variables:
                ds_current = ds_current.rename({'APCP_surface': 'tp'})

            da_precip = _compute_hourly_precip_gfs(ds_current, ds_gfs_prev, fhr)
            ds_gfs_prev = ds_current

        else:  # local
            file_ym = valid_dt.strftime("%Y-%m")
            nc_path = RAW_DIR / f"{file_ym}.nc"
            if not nc_path.exists():
                print(f"** Skipping {hr_label}: No local file {nc_path}")
                continue
            ds_local = xr.open_dataset(nc_path)
            if ("time" not in ds_local.coords) or ("tp" not in ds_local.variables):
                ds_local.close()
                print(f"** Skipping {hr_label}: 'tp' or 'time' not found.")
                continue
            target_dt64 = np.datetime64(valid_dt)
            sub_ds = ds_local.sel(time=target_dt64)
            if sub_ds.time.size == 0:
                ds_local.close()
                print(f"** Skipping {hr_label}: No data found for that time.")
                continue
            da_precip = sub_ds['tp']
            ds_local.close()

        if da_precip is None:
            print(f"** Skipping hour fhr={fhr}: da_precip not found.")
            continue

        # Ensure lat ascending, fix lon range if needed
        if 'lat' in da_precip.coords:
            da_precip = da_precip.sortby('lat', ascending=True)
        if 'lon' in da_precip.coords and da_precip['lon'].max() > 180:
            da_precip = da_precip.assign_coords(lon=(((da_precip.lon + 180) % 360) - 180)).sortby('lon')

        # Single domain-level interpolation for the coarse baseline
        coarse_domain_interp = da_precip.interp(lat=lat_global, lon=lon_global, method="linear")
        domain_coarse_2d = coarse_domain_interp.values.astype(np.float32)

        # Prepare arrays for the final model mosaic
        domain_pred_2d = np.zeros((nLat, nLon), dtype=np.float32)
        domain_wt_2d = np.zeros((nLat, nLon), dtype=np.float32)

        # --------------------------------------------
        # Inference tile-by-tile
        # --------------------------------------------
        for tid in tqdm(tile_ids, desc=f"Inference f{fhr:03d}"):
            lat_coarse, lon_coarse, lat_fine, lon_fine = tile_coordinates(tid)
            tile_model = tile_models[tid]
            elev_tile = tile_elevations[tid]

            tile_coarse = da_precip.interp(lat=lat_coarse, lon=lon_coarse, method='linear')
            c_vals = tile_coarse.values.astype(np.float32)

            lat_indices = np.searchsorted(lat_global, lat_fine)
            lon_indices = np.searchsorted(lon_global, lon_fine)

            if (not INCLUDE_ZEROS) and (c_vals.max() < PRECIP_THRESHOLD):
                fill_val_norm = -precip_mean / precip_std
                wmask = _tile_weight_mask()
                domain_pred_2d[lat_indices[0]:lat_indices[-1] + 1,
                               lon_indices[0]:lon_indices[-1] + 1] += (fill_val_norm * wmask)
                domain_wt_2d[lat_indices[0]:lat_indices[-1] + 1,
                             lon_indices[0]:lon_indices[-1] + 1] += wmask
                continue

            c_log = np.log1p(c_vals)
            c_norm = (c_log - precip_mean) / precip_std
            coarse_tensor = torch.from_numpy(c_norm).unsqueeze(0).unsqueeze(0).to(device)
            upsampled_cropped_t = _upsample_coarse_with_crop_torch(coarse_tensor, elev_tile.shape)

            elev_tile_t = torch.from_numpy(elev_tile).unsqueeze(0).unsqueeze(0).to(device)
            model_input = torch.cat([upsampled_cropped_t, elev_tile_t], dim=1)
            with torch.no_grad():
                pred_t = tile_model(model_input)
            pred_arr_norm = pred_t.squeeze().cpu().numpy()

            wmask = _tile_weight_mask()
            domain_pred_2d[lat_indices[0]:lat_indices[-1] + 1,
                           lon_indices[0]:lon_indices[-1] + 1] += (pred_arr_norm * wmask)
            domain_wt_2d[lat_indices[0]:lat_indices[-1] + 1,
                         lon_indices[0]:lon_indices[-1] + 1] += wmask

        with np.errstate(divide='ignore', invalid='ignore'):
            final_norm = domain_pred_2d / domain_wt_2d
        final_norm[~np.isfinite(final_norm)] = 0.0

        mosaic_log = (final_norm * precip_std) + precip_mean
        mosaic_mm = np.expm1(mosaic_log)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_arr = mosaic_mm / domain_coarse_2d
        ratio_arr[~np.isfinite(ratio_arr)] = np.nan

        # Accumulate totals
        domain_model_accum += np.nan_to_num(mosaic_mm, nan=0.0)
        domain_coarse_accum += np.nan_to_num(domain_coarse_2d, nan=0.0)

        # --------------------------------------------
        # 7c) Plot hour-by-hour
        # NOTE: Filenames are the same whether ratio=True or False
        # --------------------------------------------
        fig_title = f"{data_source.upper()} fhr={fhr:03d}  {hr_label}"
        out_fname = out_dir / f"inference_{data_source.lower()}_{init_datestr}_f{fhr:03d}.png"

        if ratio:
            fig, (ax_model, ax_coarse, ax_ratio) = plt.subplots(
                1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()}
            )
            for ax_ in (ax_model, ax_coarse, ax_ratio):
                ax_.set_extent([lon_global[0], lon_global[-1], lat_global[0], lat_global[-1]],
                               crs=ccrs.PlateCarree())
                ax_.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
                ax_.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
                ax_.add_feature(USCOUNTIES.with_scale('20m'), edgecolor='gray', linewidth=0.25)

            im_pred = ax_model.imshow(
                mosaic_mm, origin='lower',
                extent=[lon_global[0], lon_global[-1], lat_global[0], lat_global[-1]],
                transform=ccrs.PlateCarree(),
                cmap=cmap_hourly,
                norm=norm_hourly,
                vmin=vmin_hourly,
                vmax=vmax_hourly
            )
            ax_model.set_title(f"Downscaled Model\n{fig_title}")

            im_coarse = ax_coarse.imshow(
                domain_coarse_2d, origin='lower',
                extent=[lon_global[0], lon_global[-1], lat_global[0], lat_global[-1]],
                transform=ccrs.PlateCarree(),
                cmap=cmap_hourly,
                norm=norm_hourly,
                vmin=vmin_hourly,
                vmax=vmax_hourly
            )
            ax_coarse.set_title(f"Bilinear Coarse\n{fig_title}")

            im_ratio = ax_ratio.imshow(
                ratio_arr, origin='lower',
                extent=[lon_global[0], lon_global[-1], lat_global[0], lat_global[-1]],
                transform=ccrs.PlateCarree(),
                cmap='RdBu_r', vmin=0, vmax=2
            )
            ax_ratio.set_title(f"Ratio (Model / Coarse)\n{fig_title}")

            fig.subplots_adjust(bottom=0.1, wspace=0.25)

            # Colorbars
            if colormap.lower() == "weatherbell":
                cb_pred = fig.colorbar(
                    im_pred, ax=ax_model, orientation='horizontal',
                    fraction=0.046, pad=0.07, ticks=bounds
                )
                cb_pred.set_label("Precip (mm)")

                cb_coarse = fig.colorbar(
                    im_coarse, ax=ax_coarse, orientation='horizontal',
                    fraction=0.046, pad=0.07, ticks=bounds
                )
                cb_coarse.set_label("Precip (mm)")
            else:
                # viridis colorbars
                cb_pred = fig.colorbar(
                    im_pred, ax=ax_model, orientation='horizontal',
                    fraction=0.046, pad=0.07
                )
                cb_pred.set_label("Precip (mm)")

                cb_coarse = fig.colorbar(
                    im_coarse, ax=ax_coarse, orientation='horizontal',
                    fraction=0.046, pad=0.07
                )
                cb_coarse.set_label("Precip (mm)")

            cb_ratio = fig.colorbar(
                im_ratio, ax=ax_ratio, orientation='horizontal',
                fraction=0.046, pad=0.07
            )
            cb_ratio.set_label("Ratio")

        else:
            fig, (ax_model, ax_coarse) = plt.subplots(
                1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()}
            )
            for ax_ in (ax_model, ax_coarse):
                ax_.set_extent([lon_global[0], lon_global[-1], lat_global[0], lat_global[-1]],
                               crs=ccrs.PlateCarree())
                ax_.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
                ax_.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
                ax_.add_feature(USCOUNTIES.with_scale('20m'), edgecolor='gray', linewidth=0.25)

            im_pred = ax_model.imshow(
                mosaic_mm, origin='lower',
                extent=[lon_global[0], lon_global[-1], lat_global[0], lat_global[-1]],
                transform=ccrs.PlateCarree(),
                cmap=cmap_hourly,
                norm=norm_hourly,
                vmin=vmin_hourly,
                vmax=vmax_hourly
            )
            ax_model.set_title(f"Downscaled Model\n{fig_title}")

            im_coarse = ax_coarse.imshow(
                domain_coarse_2d, origin='lower',
                extent=[lon_global[0], lon_global[-1], lat_global[0], lat_global[-1]],
                transform=ccrs.PlateCarree(),
                cmap=cmap_hourly,
                norm=norm_hourly,
                vmin=vmin_hourly,
                vmax=vmax_hourly
            )
            ax_coarse.set_title(f"Bilinear Coarse\n{fig_title}")

            fig.subplots_adjust(bottom=0.15, wspace=0.2)

            # Single colorbar for both if "weatherbell"
            if colormap.lower() == "weatherbell":
                cbar = fig.colorbar(
                    im_coarse, ax=[ax_model, ax_coarse],
                    orientation='horizontal', fraction=0.046, pad=0.07, ticks=bounds
                )
                cbar.set_label("Precip (mm)")
            else:
                # viridis colorbar
                cbar = fig.colorbar(
                    im_coarse, ax=[ax_model, ax_coarse],
                    orientation='horizontal', fraction=0.046, pad=0.07
                )
                cbar.set_label("Precip (mm)")

        plt.savefig(out_fname, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------
    # 8) After all frames: plot the accumulated total model vs coarse
    # ------------------------------------------------------------------
    # If using viridis, we need to set our total vmax to the maximum value found in both accum arrays
    if colormap.lower() == "viridis":
        total_max_value = max(domain_model_accum.max(), domain_coarse_accum.max())
        vmin_total = 0
        vmax_total = total_max_value
    else:
        vmin_total = None
        vmax_total = None

    total_ratio_arr = None
    if ratio:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_vals = domain_model_accum / domain_coarse_accum
        ratio_vals[~np.isfinite(ratio_vals)] = np.nan
        total_ratio_arr = ratio_vals

    fig_cols = 3 if ratio else 2
    fig, axes = plt.subplots(
        1, fig_cols, figsize=(18 if ratio else 12, 6),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    axes = np.atleast_1d(axes)

    ax_model_sum = axes[0]
    ax_model_sum.set_extent([lon_global[0], lon_global[-1], lat_global[0], lat_global[-1]],
                            crs=ccrs.PlateCarree())
    ax_model_sum.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
    ax_model_sum.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    ax_model_sum.add_feature(USCOUNTIES.with_scale('20m'), edgecolor='gray', linewidth=0.25)
    im_model_sum = ax_model_sum.imshow(
        domain_model_accum,
        origin='lower',
        extent=[lon_global[0], lon_global[-1], lat_global[0], lat_global[-1]],
        transform=ccrs.PlateCarree(),
        cmap=cmap_total if colormap.lower() == "viridis" else cmap_hourly,
        norm=norm_total if colormap.lower() == "weatherbell" else None,
        vmin=vmin_total,
        vmax=vmax_total
    )
    ax_model_sum.set_title(f"Accumulated Model\n({init_datestr} + up to {hours}h)")

    ax_coarse_sum = axes[1]
    ax_coarse_sum.set_extent([lon_global[0], lon_global[-1], lat_global[0], lat_global[-1]],
                             crs=ccrs.PlateCarree())
    ax_coarse_sum.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
    ax_coarse_sum.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
    ax_coarse_sum.add_feature(USCOUNTIES.with_scale('20m'), edgecolor='gray', linewidth=0.25)
    im_coarse_sum = ax_coarse_sum.imshow(
        domain_coarse_accum,
        origin='lower',
        extent=[lon_global[0], lon_global[-1], lat_global[0], lat_global[-1]],
        transform=ccrs.PlateCarree(),
        cmap=cmap_total if colormap.lower() == "viridis" else cmap_hourly,
        norm=norm_total if colormap.lower() == "weatherbell" else None,
        vmin=vmin_total,
        vmax=vmax_total
    )
    ax_coarse_sum.set_title(f"Accumulated Coarse\n({init_datestr} + up to {hours}h)")

    if ratio:
        ax_ratio_sum = axes[2]
        ax_ratio_sum.set_extent([lon_global[0], lon_global[-1], lat_global[0], lat_global[-1]],
                                crs=ccrs.PlateCarree())
        ax_ratio_sum.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
        ax_ratio_sum.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)
        ax_ratio_sum.add_feature(USCOUNTIES.with_scale('20m'), edgecolor='gray', linewidth=0.25)
        im_ratio_sum = ax_ratio_sum.imshow(
            total_ratio_arr,
            origin='lower',
            extent=[lon_global[0], lon_global[-1], lat_global[0], lat_global[-1]],
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r',
            vmin=0,
            vmax=2
        )
        ax_ratio_sum.set_title("Accum. Ratio (Model / Coarse)")

    fig.subplots_adjust(bottom=0.15, wspace=0.25)

    # Colorbar logic for totals
    if ratio:
        # The first two axes share the same colorbar
        main_axes_for_cb = axes[:-1]
    else:
        main_axes_for_cb = axes

    if colormap.lower() == "weatherbell":
        cb_all = fig.colorbar(
            im_coarse_sum, ax=main_axes_for_cb,
            orientation='horizontal', fraction=0.046, pad=0.07, ticks=bounds
        )
        cb_all.set_label("Total Precip (mm)")
    else:
        # viridis colorbar
        cb_all = fig.colorbar(
            im_coarse_sum, ax=main_axes_for_cb,
            orientation='horizontal', fraction=0.046, pad=0.07
        )
        cb_all.set_label("Total Precip (mm)")

    if ratio:
        cb_ratio_sum = fig.colorbar(
            im_ratio_sum, ax=axes[2],
            orientation='horizontal', fraction=0.046, pad=0.07
        )
        cb_ratio_sum.set_label("Ratio")

    out_total = out_dir / f"inference_{data_source.lower()}_{init_datestr}_total.png"
    plt.savefig(out_total, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nAll inference plots (hourly + totals) saved to: {out_dir}")
    print("Done!")


if __name__ == "__main__":
    # Example usage with the new colormap option
    run_inference(
        data_source="ECMWF",
        init_datestr="2025011100",
        hours=48,
        ratio=False,
        colormap="weatherbell"
    )
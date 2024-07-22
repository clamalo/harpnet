import xarray as xr
import numpy as np
import os
import pandas as pd
import pickle


def xr_to_np(domain):
    BASE_DIR = '/Users/clamalo/documents/harpnet/load_data/'
    with open('grid_domains.pkl', 'rb') as f:
        grid_domains = pickle.load(f)
    min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx = grid_domains[domain]
    reference_ds = xr.load_dataset(os.path.join(BASE_DIR, 'reference_ds.grib2'), engine='cfgrib')
    reference_ds = reference_ds.assign_coords(longitude=(((reference_ds.longitude + 180) % 360) - 180)).sortby('longitude')
    for year in range(2020, 2021):
        if year == 1979:
            start_month = 10
        else:
            start_month = 1
        if year == 2022:
            end_month = 9
        else:
            end_month = 12
        for month in range(start_month, end_month+1):
            ds = xr.open_dataset(f'/Volumes/T9/monthly/{year}-{month:02d}.nc')
            time_index = pd.DatetimeIndex(ds.time.values)
            filtered_times = time_index[time_index.hour.isin([3, 6, 9, 12, 15, 18, 21, 0])]
            ds = ds.sel(time=filtered_times)
            ds = ds.sortby('time')
            ds['days'] = ds.time.dt.dayofyear

            cropped_ds = ds.isel(lat=slice(min_lat_idx, max_lat_idx), lon=slice(min_lon_idx, max_lon_idx))
            min_lat, max_lat, min_lon, max_lon = min(cropped_ds.lat.values), max(cropped_ds.lat.values), min(cropped_ds.lon.values), max(cropped_ds.lon.values)
            reference_ds = reference_ds.sel(latitude=slice(max_lat+0.75, min_lat-0.75), longitude=slice(min_lon-0.75, max_lon+0.75))
            input_ds = ds.interp(lat=reference_ds.latitude.values, lon=reference_ds.longitude.values)
            input_ds = input_ds.sortby('lat', ascending=True)
            ds = ds.isel(lat=slice(min_lat_idx, max_lat_idx), lon=slice(min_lon_idx, max_lon_idx))
            np.save(os.path.join(BASE_DIR, f'input_{year}_{month:02d}.npy'), input_ds.tp.values.astype('float32'))
            np.save(os.path.join(BASE_DIR, f'target_{year}_{month:02d}.npy'), ds.tp.values.astype('float32'))

xr_to_np(0)
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from datetime import datetime
import xarray as xr
import pickle
import pandas as pd
from tqdm import tqdm
import constants

def setup(domain):
    if not os.path.exists(f'{constants.domains_dir}'):
        os.makedirs(f'{constants.domains_dir}')
    if not os.path.exists(f'{constants.domains_dir}{domain}/'):
        os.makedirs(f'{constants.domains_dir}{domain}/')


def create_grid_domains():
    ds = xr.open_dataset(f'{constants.nc_dir}1979-10.nc', chunks={'time': 100})
    start_lat, start_lon = 35, -125
    end_lat, end_lon = 50, -103
    start_lat_idx = ds.lat.values.searchsorted(start_lat)
    end_lat_idx = ds.lat.values.searchsorted(end_lat)
    start_lon_idx = ds.lon.values.searchsorted(start_lon)
    end_lon_idx = ds.lon.values.searchsorted(end_lon)
    grid_domains = {}
    total_domains = 0
    for lat_idx in range(start_lat_idx, end_lat_idx, 64):
        for lon_idx in range(start_lon_idx, end_lon_idx, 64):
            grid_domains[total_domains] = [lat_idx, lat_idx+64, lon_idx, lon_idx+64]
            total_domains += 1
    with open(f'{constants.domains_dir}grid_domains.pkl', 'wb') as f:
        pickle.dump(grid_domains, f)


def xr_to_np(domain):
    print('Converting xr to np:')
    with open(f'{constants.domains_dir}grid_domains.pkl', 'rb') as f:
        grid_domains = pickle.load(f)
    min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx = grid_domains[domain]
    reference_ds = xr.load_dataset(f'{constants.base_dir}reference_ds.grib2', engine='cfgrib')
    reference_ds = reference_ds.assign_coords(longitude=(((reference_ds.longitude + 180) % 360) - 180)).sortby('longitude')
    for year in tqdm(range(2010, 2023)):
        if year == 1979:
            start_month = 10
        else:
            start_month = 1
        if year == 2022:
            end_month = 9
        else:
            end_month = 12
        for month in range(start_month, end_month+1):
            ds = xr.open_dataset(f'{constants.nc_dir}{year}-{month:02d}.nc')
            time_index = pd.DatetimeIndex(ds.time.values)
            filtered_times = time_index[time_index.hour.isin([3, 6, 9, 12, 15, 18, 21, 0])]
            ds = ds.sel(time=filtered_times)
            ds = ds.sortby('time')
            ds['days'] = ds.time.dt.dayofyear
            cropped_ds = ds.isel(lat=slice(min_lat_idx, max_lat_idx), lon=slice(min_lon_idx, max_lon_idx))
            min_lat, max_lat, min_lon, max_lon = min(cropped_ds.lat.values), max(cropped_ds.lat.values), min(cropped_ds.lon.values), max(cropped_ds.lon.values)
            reference_ds = reference_ds.sel(latitude=slice(max_lat+0.25, min_lat-0.25), longitude=slice(min_lon-0.25, max_lon+0.25))
            input_ds = ds.interp(lat=reference_ds.latitude.values, lon=reference_ds.longitude.values)
            input_ds = input_ds.sortby('lat', ascending=True)
            ds = ds.isel(lat=slice(min_lat_idx, max_lat_idx), lon=slice(min_lon_idx, max_lon_idx))

            # pre-interp
            input_ds = input_ds.interp(lat=ds.lat.values, lon=ds.lon.values)
            # pre-interp

            np.save(f'{constants.domains_dir}{domain}/input_{year}_{month:02d}.npy', input_ds.tp.values.astype('float32'))
            np.save(f'{constants.domains_dir}{domain}/target_{year}_{month:02d}.npy', ds.tp.values.astype('float32'))


def get_lats_lons(domain):
    with open(f'{constants.domains_dir}grid_domains.pkl', 'rb') as f:
        grid_domains = pickle.load(f)
    min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx = grid_domains[domain]
    ds = xr.open_dataset(f'{constants.nc_dir}1979-10.nc')
    cropped_ds = ds.isel(lat=slice(min_lat_idx, max_lat_idx), lon=slice(min_lon_idx, max_lon_idx))
    min_lat, max_lat, min_lon, max_lon = min(cropped_ds.lat.values), max(cropped_ds.lat.values), min(cropped_ds.lon.values), max(cropped_ds.lon.values)
    reference_ds = xr.load_dataset(f'{constants.base_dir}reference_ds.grib2', engine='cfgrib')
    reference_ds = reference_ds.assign_coords(longitude=(((reference_ds.longitude + 180) % 360) - 180)).sortby('longitude')
    reference_ds = reference_ds.sel(latitude=slice(max_lat+0.25, min_lat-0.25), longitude=slice(min_lon-0.25, max_lon+0.25))
    input_ds = ds.interp(lat=reference_ds.latitude.values, lon=reference_ds.longitude.values)
    input_ds = input_ds.sortby('lat', ascending=True)
    lats, lons = cropped_ds.lat.values, cropped_ds.lon.values
    input_lats, input_lons = input_ds.lat.values, input_ds.lon.values
    return lats, lons, input_lats, input_lons


def create_paths(train_test_cutoff,domain):
    train_test_cutoff = datetime.strptime(train_test_cutoff, '%Y-%m-%d:%H:%M:%S')
    input_file_paths = sorted([os.path.join(f'{constants.domains_dir}{domain}/', fp) for fp in os.listdir(f'{constants.domains_dir}{domain}/') if fp.startswith('input') and fp.endswith('.npy')], key=lambda fp: os.path.basename(fp))
    target_file_paths = sorted([os.path.join(f'{constants.domains_dir}{domain}/', fp) for fp in os.listdir(f'{constants.domains_dir}{domain}/') if fp.startswith('target') and fp.endswith('.npy')], key=lambda fp: os.path.basename(fp))
    train_input_file_paths = []
    train_target_file_paths = []
    test_input_file_paths = []
    test_target_file_paths = []
    for i, (input_fp, target_fp) in enumerate(zip(input_file_paths, target_file_paths)):
        year, month = input_fp.split('_')[-2], input_fp.split('_')[-1].split('.')[0]
        datetime_obj = datetime.strptime(f'{year}-{month}', '%Y-%m')
        if datetime_obj < train_test_cutoff:
            train_input_file_paths.append(input_fp)
            train_target_file_paths.append(target_fp)
        else:
            test_input_file_paths.append(input_fp)
            test_target_file_paths.append(target_fp)
    return train_input_file_paths, train_target_file_paths, test_input_file_paths, test_target_file_paths


class MemMapDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

def create_dataloader(input_file_paths, target_file_paths, batch_size=64, shuffle=True):
    input_arr = np.concatenate([np.load(fp, mmap_mode='r') for fp in input_file_paths], axis=0)
    target_arr = np.concatenate([np.load(fp, mmap_mode='r') for fp in target_file_paths], axis=0)
    dataset = MemMapDataset(input_arr, target_arr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, dataloader


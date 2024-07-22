from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import xarray as xr
import pickle
import pandas as pd
from tqdm import tqdm
import constants
import matplotlib.pyplot as plt
import cartopy
from metpy.plots import USCOUNTIES
import matplotlib.patches as patches


def setup(domain):
    if not os.path.exists(f'{constants.domains_dir}'):
        os.makedirs(f'{constants.domains_dir}')
    if not os.path.exists(f'{constants.domains_dir}/{domain}/'):
        os.makedirs(f'{constants.domains_dir}/{domain}/')


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


def xr_to_np(domain, first_month, last_month):
    first_month = datetime(first_month[0], first_month[1], 1)
    last_month = datetime(last_month[0], last_month[1], 1)
    print('Converting xr to np...')
    with open(f'{constants.domains_dir}grid_domains.pkl', 'rb') as f:
        grid_domains = pickle.load(f)
    min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx = grid_domains[domain]
    reference_ds = xr.load_dataset(f'{constants.base_dir}reference_ds.grib2', engine='cfgrib')
    reference_ds = reference_ds.assign_coords(longitude=(((reference_ds.longitude + 180) % 360) - 180)).sortby('longitude')

    total_months = (last_month.year - first_month.year) * 12 + last_month.month - first_month.month + 1

    current_month = first_month
    for _ in tqdm(range(total_months), desc="Processing months"):
        year, month = current_month.year, current_month.month
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
        current_month += relativedelta(months=1)


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


def create_paths(domain, first_month, last_month, train_test_cutoff):
    train_test_cutoff = datetime.strptime(train_test_cutoff, '%Y-%m-%d:%H:%M:%S')
    first_month = datetime(first_month[0], first_month[1], 1)
    last_month = datetime(last_month[0], last_month[1], 1)

    input_file_paths = []
    target_file_paths = []

    current_month = first_month
    while current_month <= last_month:
        input_fp = f'{constants.domains_dir}{domain}/input_{current_month.year}_{current_month.month:02d}.npy'
        target_fp = f'{constants.domains_dir}{domain}/target_{current_month.year}_{current_month.month:02d}.npy'
        input_file_paths.append(input_fp)
        target_file_paths.append(target_fp)
        current_month += relativedelta(months=1)
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
    def load_files_in_batches(file_paths, batch_size=64):
        arrays = []
        for i in tqdm(range(0, len(file_paths), batch_size)):
            batch_paths = file_paths[i:i + batch_size]
            batch_arrays = [np.load(fp, mmap_mode='r') for fp in batch_paths]
            arrays.append(np.concatenate(batch_arrays, axis=0))
        return np.concatenate(arrays, axis=0)
    input_arr = load_files_in_batches(input_file_paths, batch_size=batch_size)
    target_arr = load_files_in_batches(target_file_paths, batch_size=batch_size)
    dataset = MemMapDataset(input_arr, target_arr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, dataloader


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def test(domain, model, dataloader, criterion, device):
    lats, lons, input_lats, input_lons = get_lats_lons(domain)
    model.eval()
    running_loss = 0.0
    random_10 = np.random.randint(0, len(dataloader), 10)

    plotted = 0
    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, targets)
        running_loss += loss.item()

        if i in random_10:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': cartopy.crs.PlateCarree()})
            axs[0].pcolormesh(lons, lats, inputs[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            axs[1].pcolormesh(lons, lats, outputs[0].cpu().detach().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            axs[2].pcolormesh(lons, lats, targets[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            for ax in axs:
                ax.coastlines()
                ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray')
            box = patches.Rectangle((lons[0], lats[0]), lons[-1] - lons[0], lats[-1] - lats[0],
                                    linewidth=1, edgecolor='r', facecolor='none')
            plt.savefig(f'figures/test/{plotted}.png')
            plt.close()
            plotted += 1

    return running_loss / len(dataloader)
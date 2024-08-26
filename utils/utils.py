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
import torch
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def setup(domain):
    if not os.path.exists(f'{constants.domains_dir}'):
        os.makedirs(f'{constants.domains_dir}')
    if not os.path.exists(f'{constants.domains_dir}/{domain}/'):
        os.makedirs(f'{constants.domains_dir}/{domain}/')
    if not os.path.exists(f'{constants.checkpoints_dir}'):
        os.makedirs(f'{constants.checkpoints_dir}')



def create_grid_domains():
    start_lat, start_lon = 30, -125
    end_lat, end_lon = 55, -100
    grid_domains = {}
    total_domains = 0
    for lat in range(start_lat, end_lat, 4):
        for lon in range(start_lon, end_lon, 4):
            grid_domains[total_domains] = [lat, lat + 4, lon, lon + 4]
            total_domains += 1
    with open(f'{constants.domains_dir}grid_domains.pkl', 'wb') as f:
        pickle.dump(grid_domains, f)
        
        
def create_grid_domains():
    start_lat, start_lon = 41, -93
    end_lat, end_lon = 49, -82
    grid_domains = {}
    total_domains = 0
    for lat in range(start_lat, end_lat, 4):
        for lon in range(start_lon, end_lon, 4):
            grid_domains[total_domains] = [lat, lat + 4, lon, lon + 4]
            total_domains += 1
    with open(f'{constants.domains_dir}grid_domains.pkl', 'wb') as f:
        pickle.dump(grid_domains, f)



def scale_coordinates(x, scale_factor):
    resolution = x[1] - x[0]
    fine_resolution = resolution / scale_factor
    x_fine = []
    for center in x:
        start = center - (resolution / 2) + (fine_resolution / 2)
        fine_points = [start + i * fine_resolution for i in range(scale_factor)]
        x_fine.extend(fine_points)
    return np.array(x_fine)



def xr_to_np(domain, first_month, last_month, pad=False):
    first_month = datetime(first_month[0], first_month[1], 1)
    last_month = datetime(last_month[0], last_month[1], 1)
    with open(f'{constants.domains_dir}grid_domains.pkl', 'rb') as f:
        grid_domains = pickle.load(f)
    min_lat, max_lat, min_lon, max_lon = grid_domains[domain]

    reference_ds = xr.load_dataset(f'{constants.base_dir}reference_ds.grib2', engine='cfgrib')
    reference_ds = reference_ds.assign_coords(longitude=(((reference_ds.longitude + 180) % 360) - 180)).sortby('longitude')
    reference_ds = reference_ds.sortby('latitude', ascending=True)

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

        cropped_reference_ds = reference_ds.sel(latitude=slice(min_lat, max_lat-0.25), longitude=slice(min_lon, max_lon-0.25))
        cropped_reference_ds_latitudes = cropped_reference_ds.latitude.values
        cropped_reference_ds_longitudes = cropped_reference_ds.longitude.values
        if pad:
            cropped_input_reference_ds = reference_ds.sel(latitude=slice(min_lat-0.25, max_lat), longitude=slice(min_lon-0.25, max_lon))
            cropped_input_reference_ds_latitudes = cropped_input_reference_ds.latitude.values
            cropped_input_reference_ds_longitudes = cropped_input_reference_ds.longitude.values
        
        fine_lats = scale_coordinates(cropped_reference_ds_latitudes, constants.scale_factor)
        fine_lons = scale_coordinates(cropped_reference_ds_longitudes, constants.scale_factor)

        fine_ds = ds.interp(lat=fine_lats, lon=fine_lons)
        if pad:
            coarse_ds = ds.interp(lat=cropped_input_reference_ds_latitudes, lon=cropped_input_reference_ds_longitudes)
        else:
            coarse_ds = ds.interp(lat=cropped_reference_ds_latitudes, lon=cropped_reference_ds_longitudes)

        np.save(f'{constants.domains_dir}{domain}/input_{year}_{month:02d}.npy', coarse_ds.tp.values.astype('float32'))
        np.save(f'{constants.domains_dir}{domain}/target_{year}_{month:02d}.npy', fine_ds.tp.values.astype('float32'))
        current_month += relativedelta(months=1)



def get_lats_lons(domain, pad):
    with open(f'{constants.domains_dir}grid_domains.pkl', 'rb') as f:
        grid_domains = pickle.load(f)

    min_lat, max_lat, min_lon, max_lon = grid_domains[domain]
    
    reference_ds = xr.load_dataset(f'{constants.base_dir}reference_ds.grib2', engine='cfgrib')
    reference_ds = reference_ds.assign_coords(longitude=(((reference_ds.longitude + 180) % 360) - 180)).sortby('longitude')
    reference_ds = reference_ds.sortby('latitude', ascending=True)

    cropped_reference_ds = reference_ds.sel(latitude=slice(min_lat, max_lat-0.25), longitude=slice(min_lon, max_lon-0.25))
    cropped_reference_ds_latitudes = cropped_reference_ds.latitude.values
    cropped_reference_ds_longitudes = cropped_reference_ds.longitude.values
    if pad:
        cropped_input_reference_ds = reference_ds.sel(latitude=slice(min_lat-0.25, max_lat), longitude=slice(min_lon-0.25, max_lon))
        cropped_input_reference_ds_latitudes = cropped_input_reference_ds.latitude.values
        cropped_input_reference_ds_longitudes = cropped_input_reference_ds.longitude.values
    
    fine_lats = scale_coordinates(cropped_reference_ds_latitudes, constants.scale_factor)
    fine_lons = scale_coordinates(cropped_reference_ds_longitudes, constants.scale_factor)

    if pad:
        return fine_lats, fine_lons, cropped_input_reference_ds_latitudes, cropped_input_reference_ds_longitudes
    else:
        return fine_lats, fine_lons, cropped_reference_ds_latitudes, cropped_reference_ds_longitudes


def create_paths(domain, first_month, last_month, train_test):
    if type(train_test) == str:
        train_test = datetime.strptime(train_test, '%Y-%m-%d:%H:%M:%S')
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

    if type(train_test) == str:
        train_input_file_paths = []
        train_target_file_paths = []
        test_input_file_paths = []
        test_target_file_paths = []
        for i, (input_fp, target_fp) in enumerate(zip(input_file_paths, target_file_paths)):
            year, month = input_fp.split('_')[-2], input_fp.split('_')[-1].split('.')[0]
            datetime_obj = datetime.strptime(f'{year}-{month}', '%Y-%m')
            if datetime_obj < train_test:
                train_input_file_paths.append(input_fp)
                train_target_file_paths.append(target_fp)
            else:
                test_input_file_paths.append(input_fp)
                test_target_file_paths.append(target_fp)

    elif type(train_test) == float:
        indices = list(range(len(input_file_paths)))
        random.shuffle(indices)
        train_indices = indices[int(len(indices) * train_test):]
        test_indices = indices[:int(len(indices) * train_test)]
        train_input_file_paths = [input_file_paths[i] for i in train_indices]
        train_target_file_paths = [target_file_paths[i] for i in train_indices]
        test_input_file_paths = [input_file_paths[i] for i in test_indices]
        test_target_file_paths = [target_file_paths[i] for i in test_indices]

    return train_input_file_paths, train_target_file_paths, test_input_file_paths, test_target_file_paths


class MemMapDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

def create_dataloader(input_file_paths, target_file_paths, batch_size=32, shuffle=True):
    def load_files_in_batches(file_paths, batch_size=32):
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


def train(domain, model, dataloader, criterion, optimizer, device, pad=False, plot=False):
    model.train()
    losses = []

    if plot:
        lats, lons, input_lats, input_lons = get_lats_lons(domain, pad)
        random_10 = np.random.choice(range(len(dataloader)), 10, replace=False)
        plotted = 0

    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if plot and i in random_10:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': cartopy.crs.PlateCarree()})
            axs[0].pcolormesh(input_lons, input_lats, inputs[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            axs[1].pcolormesh(lons, lats, outputs[0].cpu().detach().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            axs[2].pcolormesh(lons, lats, targets[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            for ax in axs:
                ax.coastlines()
                ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray')
            box = patches.Rectangle((lons[0], lats[0]), lons[-1] - lons[0], lats[-1] - lats[0],
                                    linewidth=1, edgecolor='r', facecolor='none')
            axs[0].add_patch(box)
            plt.suptitle(f'Loss: {criterion(outputs[0], targets[0]).item():.3f}')
            plt.savefig(f'figures/train/{plotted}.png')
            plt.close()
            plotted += 1

    return np.mean(losses)


def test(domain, model, dataloader, criterion, device, pad=False, plot=True):
    model.eval()
    losses = []
    bilinear_losses = []

    if plot:
        lats, lons, input_lats, input_lons = get_lats_lons(domain, pad)
        random_10 = np.random.choice(range(len(dataloader)), 10, replace=False)
        plotted = 0

    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        
        cropped_inputs = inputs[:, 1:-1, 1:-1]
        interpolated_inputs = torch.nn.functional.interpolate(cropped_inputs.unsqueeze(1), size=(64, 64), mode='bilinear').squeeze(1)
        
        bilinear_loss = criterion(interpolated_inputs, targets)

        losses.append(loss.item())
        bilinear_losses.append(bilinear_loss.item())

        if plot and i in random_10:
            fig, axs = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={'projection': cartopy.crs.PlateCarree()})
            axs[0].pcolormesh(input_lons, input_lats, inputs[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            axs[1].pcolormesh(lons, lats, interpolated_inputs[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            axs[2].pcolormesh(lons, lats, outputs[0].cpu().detach().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            axs[3].pcolormesh(lons, lats, targets[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=10)
            for ax in axs:
                ax.coastlines()
                ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray')
            box = patches.Rectangle((lons[0], lats[0]), lons[-1] - lons[0], lats[-1] - lats[0],
                                    linewidth=1, edgecolor='r', facecolor='none')
            axs[0].add_patch(box)
            axs[0].set_title('Input')
            axs[1].set_title('Bilinear')
            axs[2].set_title('HARPNET Output')
            axs[3].set_title('Target')
            plt.suptitle(f'Loss: {criterion(outputs[0], targets[0]).item():.3f}')
            plt.savefig(f'figures/test/{plotted}.png')
            plt.close()
            plotted += 1

    return np.mean(losses), np.mean(bilinear_losses)


def sort_epochs(patches=None):
    if not os.path.exists(f'{constants.checkpoints_dir}best/'):
        os.makedirs(f'{constants.checkpoints_dir}best/')

    if patches is None:
        patches = [f for f in os.listdir(constants.checkpoints_dir) if f != 'best' and f != '.DS_Store']

    for patch in patches:
        checkpoint_dir = f'{constants.checkpoints_dir}{patch}/'

        # Get all checkpoint files in the directory
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('_model.pt')]

        if len(checkpoint_files) == 0:
            continue

        best_members = []

        for checkpoint_file in tqdm(checkpoint_files):
            # Extract the epoch number from the filename
            epoch = int(checkpoint_file.split('_')[0])
            
            checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_file))
            train_loss = checkpoint['train_loss']
            test_loss = checkpoint['test_loss']
            best_members.append((epoch, train_loss, test_loss))

        # Sort by test_loss (ascending) and select the top 5
        best_members = sorted(best_members, key=lambda x: x[2])[:5]

        best_member = best_members[0]
        best_checkpoint = torch.load(os.path.join(checkpoint_dir, f'{best_member[0]}_model.pt'))
        torch.save(best_checkpoint, os.path.join(f'{constants.checkpoints_dir}best/', f'{patch}_model.pt'))
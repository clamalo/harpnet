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
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call this function at the beginning of your script or before creating your model
set_seed(42)


def setup(domain):
    if not os.path.exists(f'{constants.domains_dir}'):
        os.makedirs(f'{constants.domains_dir}')
    if not os.path.exists(f'{constants.domains_dir}/{domain}/'):
        os.makedirs(f'{constants.domains_dir}/{domain}/')
    if not os.path.exists(f'checkpoints/{domain}/'):
        os.makedirs(f'checkpoints/{domain}/')


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


def crop_reference_ds(reference_ds, cropped_ds):
    max_lat, min_lat, max_lon, min_lon = max(cropped_ds.lat.values), min(cropped_ds.lat.values), max(cropped_ds.lon.values), min(cropped_ds.lon.values)
    def crop_ds(reference_ds, min_lat, max_lat, min_lon, max_lon):
        return reference_ds.sel(latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon))
    cropped_reference_ds = crop_ds(reference_ds, min_lat, max_lat, min_lon, max_lon)
    min_lat -= 0.25 * ((cropped_reference_ds.latitude.values[0] - cropped_ds.lat.values[0]) > 0.125)
    max_lat += 0.25 * ((cropped_ds.lat.values[-1] - cropped_reference_ds.latitude.values[-1]) > 0.125)
    min_lon -= 0.25 * ((cropped_reference_ds.longitude.values[0] - cropped_ds.lon.values[0]) > 0.125)
    max_lon += 0.25 * ((cropped_ds.lon.values[-1] - cropped_reference_ds.longitude.values[-1]) > 0.125)
    cropped_reference_ds = crop_ds(reference_ds, min_lat, max_lat, min_lon, max_lon)
    return cropped_reference_ds


def xr_to_np(domain, first_month, last_month):
    first_month = datetime(first_month[0], first_month[1], 1)
    last_month = datetime(last_month[0], last_month[1], 1)
    with open(f'{constants.domains_dir}grid_domains.pkl', 'rb') as f:
        grid_domains = pickle.load(f)
    min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx = grid_domains[domain]
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

        cropped_ds = ds.isel(lat=slice(min_lat_idx, max_lat_idx), lon=slice(min_lon_idx, max_lon_idx))
        min_lat, max_lat, min_lon, max_lon = min(cropped_ds.lat.values), max(cropped_ds.lat.values), min(cropped_ds.lon.values), max(cropped_ds.lon.values)
        
        reference_ds = crop_reference_ds(reference_ds, cropped_ds)

        input_ds = ds.interp(lat=reference_ds.latitude.values, lon=reference_ds.longitude.values)
        input_ds = input_ds.sortby('lat', ascending=True)
        ds = ds.isel(lat=slice(min_lat_idx, max_lat_idx), lon=slice(min_lon_idx, max_lon_idx))
        # ds = ds.coarsen(lat=2, lon=2).mean() #if coarsen
        # input_ds = input_ds.interp(lat=ds.lat.values, lon=ds.lon.values) #if pre-interp

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
    reference_ds = reference_ds.sortby('latitude', ascending=True)
    
    reference_ds = crop_reference_ds(reference_ds, cropped_ds)

    input_ds = ds.interp(lat=reference_ds.latitude.values, lon=reference_ds.longitude.values)
    input_ds = input_ds.sortby('lat', ascending=True)
    # cropped_ds = cropped_ds.coarsen(lat=2, lon=2).mean() #if coarsen
    # input_ds = input_ds.interp(lat=cropped_ds.lat.values, lon=cropped_ds.lon.values) #if pre-interp
    lats, lons = cropped_ds.lat.values, cropped_ds.lon.values
    input_lats, input_lons = input_ds.lat.values, input_ds.lon.values

    return lats, lons, input_lats, input_lons


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
    

def create_dataloader(input_file_paths, target_file_paths, batch_size=64, shuffle=True):
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


def train(domain, model, dataloader, criterion, optimizer, device, plot=False):
    model.train()
    losses = []

    if plot:
        lats, lons, input_lats, input_lons = get_lats_lons(domain)
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


def test(domain, model, dataloader, criterion, device):
    lats, lons, input_lats, input_lons = get_lats_lons(domain)
    model.eval()
    losses = []

    random_10 = np.random.choice(range(len(dataloader)), 10, replace=False)
    plotted = 0

    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())

        if i in random_10:
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
            plt.savefig(f'figures/test/{plotted}.png')
            plt.close()
            plotted += 1

    return np.mean(losses)




def train_gan(domain, generator, discriminator, dataloader, criterion_GAN, criterion_content, optimizer_G, optimizer_D, device, plot=False):
    generator.train()
    discriminator.train()
    G_losses = []
    D_losses = []

    if plot:
        lats, lons, input_lats, input_lons = get_lats_lons(domain)
        random_10 = np.random.choice(range(len(dataloader)), 10, replace=False)
        plotted = 0

    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets = inputs.to(device)/10, targets.to(device)/10

        # Train Discriminator
        optimizer_D.zero_grad()
        real_labels = torch.ones(inputs.size(0), 1, device=device)
        fake_labels = torch.zeros(inputs.size(0), 1, device=device)

        outputs = generator(inputs)
        real_output = discriminator(targets).view(-1, 1)
        fake_output = discriminator(outputs.detach()).view(-1, 1)

        d_loss_real = criterion_GAN(real_output, real_labels)
        d_loss_fake = criterion_GAN(fake_output, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_output = discriminator(outputs).view(-1, 1)
        g_loss_adv = criterion_GAN(fake_output, real_labels)
        g_loss_content = criterion_content(outputs, targets)
        g_loss = g_loss_adv + g_loss_content
        g_loss.backward()
        optimizer_G.step()

        print(f'G Loss: {g_loss.item()} - D Loss: {d_loss.item()}')

        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        if plot and i in random_10:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': cartopy.crs.PlateCarree()})
            axs[0].pcolormesh(input_lons, input_lats, inputs[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=1)
            axs[1].pcolormesh(lons, lats, outputs[0].cpu().detach().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=1)
            axs[2].pcolormesh(lons, lats, targets[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=1)
            for ax in axs:
                ax.coastlines()
                ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray')
            box = patches.Rectangle((lons[0], lats[0]), lons[-1] - lons[0], lats[-1] - lats[0],
                                    linewidth=1, edgecolor='r', facecolor='none')
            axs[0].add_patch(box)
            plt.suptitle(f'G Loss: {g_loss.item():.3f} - D Loss: {d_loss.item():.3f}')
            plt.savefig(f'figures/train/{plotted}.png')
            plt.close()
            plotted += 1

    return np.mean(G_losses), np.mean(D_losses)

def test_gan(domain, generator, dataloader, criterion, device):
    lats, lons, input_lats, input_lons = get_lats_lons(domain)
    generator.eval()
    losses = []

    random_10 = np.random.choice(range(len(dataloader)), 10, replace=False)
    plotted = 0

    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets = inputs.to(device)/10, targets.to(device)/10
        outputs = generator(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())

        if i in random_10:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': cartopy.crs.PlateCarree()})
            axs[0].pcolormesh(input_lons, input_lats, inputs[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=1)
            axs[1].pcolormesh(lons, lats, outputs[0].cpu().detach().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=1)
            axs[2].pcolormesh(lons, lats, targets[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=1)
            for ax in axs:
                ax.coastlines()
                ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray')
            box = patches.Rectangle((lons[0], lats[0]), lons[-1] - lons[0], lats[-1] - lats[0],
                                    linewidth=1, edgecolor='r', facecolor='none')
            axs[0].add_patch(box)
            plt.suptitle(f'Loss: {criterion(outputs[0], targets[0]).item():.3f}')
            plt.savefig(f'figures/test/{plotted}.png')
            plt.close()
            plotted += 1

    return np.mean(losses)
import torch.nn as nn
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
from datetime import datetime
from dask.diagnostics import ProgressBar
import torch.nn.functional as F
import random
import pickle
from utils.model import UNetWithAttention
from utils.utils import *
import warnings
warnings.filterwarnings("ignore")



BASE_DIR = '/Users/clamalo/documents/harpnet/data/load_data/'
LOAD = True
months = -1
max_lat, min_lat, max_lon, min_lon = 41.75, 36.225, -104.25, -109.825


if LOAD:
    #LOAD TP DATA
    summed_dir = '/Volumes/T9/monthly/'
    file_paths = [os.path.join(summed_dir, fp) for fp in os.listdir(summed_dir) if fp.endswith('.nc')]
    file_paths = sorted(file_paths, key=lambda fp: os.path.basename(fp))[:15]

    with ProgressBar():
        ds = xr.open_mfdataset(file_paths, combine='by_coords', parallel=True, chunks={'time': 100})
        ds = ds.assign_coords(hour=ds.time.dt.hour, day=ds.time.dt.dayofyear)
    ds = ds.sortby('time')
    ds['days'] = ds.time.dt.dayofyear
    for var in ds.data_vars:
        ds[var] = ds[var].astype('float32')

    reference_ds = xr.load_dataset(os.path.join(BASE_DIR, 'reference_ds.grib2'), engine='cfgrib')
    reference_ds = reference_ds.assign_coords(longitude=(((reference_ds.longitude + 180) % 360) - 180)).sortby('longitude')
    reference_ds = reference_ds.sel(latitude=slice(max_lat+0.5, min_lat-0.5), longitude=slice(min_lon-0.5, max_lon+0.5))

    input_ds = ds.interp(lat=reference_ds.latitude.values, lon=reference_ds.longitude.values)

    # #crop both datasets to before october 1 2020 0z (not inclusive) THIS IS THE TRAIN/TEST SPLIT
    # input_ds = input_ds.sel(time=slice(None, np.datetime64('2020-09-30T21:00:00')))
    # ds = ds.sel(time=slice(None, np.datetime64('2020-09-30T21:00:00')))

    ds = ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

    shapes = {}
    shapes['input_latitudes'] = input_ds.lat.values
    shapes['input_longitudes'] = input_ds.lon.values
    shapes['output_latitudes'] = ds.lat.values
    shapes['output_longitudes'] = ds.lon.values

    indices = list(range(input_ds.time.shape[0]))
    random.shuffle(indices)
    

    for var in input_ds.data_vars:
        load_array(input_ds[var], shapes, indices, BASE_DIR)
    load_array(ds['tp'], shapes, indices, BASE_DIR, target=True)
    load_array(ds['days'], shapes, indices, BASE_DIR)

    with open(os.path.join(BASE_DIR, 'shapes.pkl'), 'wb') as f:
        pickle.dump(shapes, f)



variables = ['tp']


if not LOAD:
    with open(os.path.join(BASE_DIR, 'shapes.pkl'), 'rb') as f:
        shapes = pickle.load(f)
print(shapes.keys())
input_ds_latitudes, input_ds_longitudes = shapes['input_latitudes'], shapes['input_longitudes']
ds_latitudes, ds_longitudes = shapes['output_latitudes'], shapes['output_longitudes']

train_var_dict = {var: np.memmap(os.path.join(BASE_DIR, f"{var}_mmap_train.dat"), dtype='float32', mode='r', shape=shapes[f'{var}_train']) for var in variables}
train_data = np.concatenate([train_var_dict[var] for var in variables], axis=1)
train_labels = np.memmap(os.path.join(BASE_DIR, 'target_mmap_train.dat'), dtype='float32', mode='r', shape=shapes['target_train'])
train_days = np.memmap(os.path.join(BASE_DIR, 'days_mmap_train.dat'), dtype='float32', mode='r', shape=shapes['days_train'])

train_dataset = MemMapDataset(train_data, train_labels, train_days)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

print(f'({len(train_dataloader)}, 32, {next(iter(train_dataloader))[0].shape[1]}, {next(iter(train_dataloader))[0].shape[2]}, {next(iter(train_dataloader))[0].shape[3]})')

model = UNetWithAttention(1, 1)
device = torch.device('mps')
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_params = sum(p.numel() for p in model.parameters())
print(f'\nNumber of parameters: {num_params/1e6:.2f}M')


# TRAIN
num_epochs = 51
for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        labels = batch[1].to(device)
        data = batch[0].to(device)

        optimizer.zero_grad()
        outputs = model(data)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_dataloader)}')

    # torch.save(model.state_dict(), 'train1.pt')
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss/len(train_dataloader)
    }
    torch.save(checkpoint, f'checkpoints/train_e{epoch}.pt')
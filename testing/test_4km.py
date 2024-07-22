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
import pandas as pd
import cftime


resolution = 4



BASE_DIR = '/Users/clamalo/documents/harpnet/load_data/'
LOAD = True
if resolution == 4:
    max_lat, min_lat, max_lon, min_lon = 44.85, 39.3, -108.7, -114.25
else:
    max_lat, min_lat, max_lon, min_lon = 45, 35, -105, -115

if LOAD:
    #LOAD TP DATA
    summed_dir = '/Volumes/T9/monthly/'
    file_paths = [os.path.join(summed_dir, fp) for fp in os.listdir(summed_dir) if fp.endswith('.nc')]
    file_paths = sorted(file_paths, key=lambda fp: os.path.basename(fp))

    with ProgressBar():
        ds = xr.open_mfdataset(file_paths, combine='by_coords', parallel=True, chunks={'time': 100})
        time_index = pd.DatetimeIndex(ds.time.values)
        filtered_times = time_index[time_index.hour.isin([3, 6, 9, 12, 15, 18, 21, 0])]
        ds = ds.sel(time=filtered_times)

    ds = ds.sortby('time')
    ds['days'] = ds.time.dt.dayofyear
    for var in ds.data_vars:
        ds[var] = ds[var].astype('float32')

    reference_ds = xr.load_dataset(os.path.join(BASE_DIR, 'reference_ds.grib2'), engine='cfgrib')
    reference_ds = reference_ds.assign_coords(longitude=(((reference_ds.longitude + 180) % 360) - 180)).sortby('longitude')
    reference_ds = reference_ds.sel(latitude=slice(max_lat+0.5, min_lat-0.5), longitude=slice(min_lon-0.5, max_lon+0.5))

    input_ds = ds.interp(lat=reference_ds.latitude.values, lon=reference_ds.longitude.values)

    #crop both datasets to before october 1 2020 0z (not inclusive) THIS IS THE TRAIN/TEST SPLIT
    input_ds = input_ds.sel(time=slice(np.datetime64('2020-10-01T00:00:00'), None))
    ds = ds.sel(time=slice(np.datetime64('2020-10-01T00:00:00'), None))
    input_ds = input_ds.sel(time=slice(np.datetime64('2022-09-25T00:00:00'), None))
    ds = ds.sel(time=slice(np.datetime64('2022-09-25T00:00:00'), None))

    ds = ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

    shapes = {}
    shapes['input_latitudes'] = input_ds.lat.values
    shapes['input_longitudes'] = input_ds.lon.values
    shapes['output_latitudes'] = ds.lat.values
    shapes['output_longitudes'] = ds.lon.values

    indices = list(range(input_ds.time.shape[0]))
    random.shuffle(indices)

    print(input_ds[var].shape, input_ds['tp'].shape, ds['tp'].shape, ds['days'].shape)

    for var in input_ds.data_vars:
        load_array_test(input_ds[var], shapes, indices, BASE_DIR)
    load_array_test(ds['tp'], shapes, indices, BASE_DIR, target=True)
    load_array_test(ds['days'], shapes, indices, BASE_DIR)

    with open(os.path.join(BASE_DIR, 'shapes.pkl'), 'wb') as f:
        pickle.dump(shapes, f)

variables = ['tp']

if not LOAD:
    with open(os.path.join(BASE_DIR, 'shapes.pkl'), 'rb') as f:
        shapes = pickle.load(f)
input_ds_latitudes, input_ds_longitudes = shapes['input_latitudes'], shapes['input_longitudes']
ds_latitudes, ds_longitudes = shapes['output_latitudes'], shapes['output_longitudes']

test_var_dict = {var: np.memmap(os.path.join(BASE_DIR, f"{var}_mmap_test.dat"), dtype='float32', mode='r', shape=shapes[f'{var}_test']) for var in variables}
test_data = np.concatenate([test_var_dict[var] for var in variables], axis=1)
test_labels = np.memmap(os.path.join(BASE_DIR, 'target_mmap_test.dat'), dtype='float32', mode='r', shape=shapes['target_test'])
test_days = np.memmap(os.path.join(BASE_DIR, 'days_mmap_test.dat'), dtype='float32', mode='r', shape=shapes['days_test'])
test_dataset = MemMapDataset(test_data, test_labels, test_days)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f'({len(test_dataloader)}, 32, {next(iter(test_dataloader))[0].shape[1]}, {next(iter(test_dataloader))[0].shape[2]}, {next(iter(test_dataloader))[0].shape[3]})')

quit()

model = UNetWithAttention(1, 1).to('mps').eval()
criterion = nn.MSELoss()
for epoch in range(0,91):
    original_checkpoint = torch.load(f'checkpoints/train_e{epoch}.pt')
    model_weights = original_checkpoint['model_state_dict']
    train_loss = original_checkpoint['train_loss']
    
    model.load_state_dict(model_weights)

    test_loss = 0
    winter_loss = 0
    spring_loss = 0
    summer_loss = 0
    fall_loss = 0
    with torch.no_grad():
        for i, (data, labels, days) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            data, labels, days = data.to('mps'), labels.to('mps'), days.to('mps')
            output = model(data)
            loss = criterion(output, labels)
            test_loss += loss.item()

            winter_loss += criterion(output[(days < 60) | (days > 335)], labels[(days < 60) | (days > 335)]).item()
            spring_loss += criterion(output[(60 <= days) & (days < 152)], labels[(60 <= days) & (days < 152)]).item()
            summer_loss += criterion(output[(152 <= days) & (days < 244)], labels[(152 <= days) & (days < 244)]).item()
            fall_loss += criterion(output[(244 <= days) & (days <= 335)], labels[(244 <= days) & (days <= 335)]).item()


    test_loss /= len(test_dataloader)
    winter_loss /= len(test_dataloader)
    spring_loss /= len(test_dataloader)
    summer_loss /= len(test_dataloader)
    fall_loss /= len(test_dataloader)

    print(f'Epoch {epoch} Test Loss: {test_loss}')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': original_checkpoint['model_state_dict'],
        'optimizer_state_dict': original_checkpoint['optimizer_state_dict'],
        'train_loss': train_loss,
        'test_loss': test_loss,
        'winter_test_loss': winter_loss,
        'spring_test_loss': spring_loss,
        'summer_test_loss': summer_loss,
        'fall_test_loss': fall_loss,
    }
    torch.save(checkpoint, f'checkpoints/train_e{epoch}.pt')
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm 
from scipy.ndimage import binary_dilation
import torch
import torch.nn.functional as F
from ecmwf.opendata import Client
client = Client("ecmwf", beta=True)
import requests
import os
import pickle
from ecmwf.opendata import Client
from metpy.plots import USCOUNTIES
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from utils import *
from model import UNetWithAttention



datestr = '20240109'
cycle = '12'
ingest = False 
ecmwf = False
max_lat, min_lat, max_lon, min_lon = 42.05, 36.5, -119.025, -124.6


with open(os.path.join('/Users/clamalo/documents/harpnet/load_data/shapes.pkl'), 'rb') as f:
    shapes = pickle.load(f)
ds_latitudes = shapes['output_latitudes']
ds_longitudes = shapes['output_longitudes']


gfs_datasets = []
raw_ecmwf_datasets = []
ecmwf_datasets = []
for frame in range(3,73,3):

    if ingest:
        idx_url = f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{datestr}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{frame:03d}.idx'
        start_byte = requests.get(idx_url).text.split('\n')[595].split(':')[1]
        end_byte = requests.get(idx_url).text.split('\n')[596].split(':')[1]
        gfs_url = f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{datestr}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{frame:03d}'
        r = requests.get(gfs_url, headers={'Range': f'bytes={start_byte}-{end_byte}'}, allow_redirects=True)
        open(f'data/gfs.f{frame:03d}.grib', 'wb').write(r.content)

        if ecmwf:
            client = Client("ecmwf", beta=True)
            parameters = ['tp']
            client.retrieve(
                date=0,
                time=12,
                step=frame,
                stream="oper",
                type="fc",
                levtype="sfc",
                param=parameters,
                target=f'data/ecmwf.f{frame:03d}.grib'
            )


    gfs_ds = xr.open_dataset(f'data/gfs.f{frame:03d}.grib', engine='cfgrib')
    gfs_ds = gfs_ds.assign_coords(longitude=(((gfs_ds.longitude + 180) % 360) - 180)).sortby('longitude')
    gfs_ds = gfs_ds.sortby('latitude', ascending=False)
    gfs_ds = gfs_ds.sel(latitude=slice(max_lat+0.5, min_lat-0.5), longitude=slice(min_lon-0.5, max_lon+0.5))

    if ecmwf:
        ecmwf_ds = xr.open_dataset(f'data/ecmwf.f{frame:03d}.grib', engine='cfgrib')
        ecmwf_ds = ecmwf_ds.assign_coords(longitude=(((ecmwf_ds.longitude + 180) % 360) - 180)).sortby('longitude')
        ecmwf_ds = ecmwf_ds.sortby('latitude', ascending=False)
        ecmwf_ds = ecmwf_ds.sel(latitude=slice(max_lat+0.5, min_lat-0.5), longitude=slice(min_lon-0.5, max_lon+0.5))

    for var in gfs_ds.data_vars:
        gfs_ds[var] = gfs_ds[var].astype('float32')
    if frame % 6 == 0:
        gfs_ds['tp'] = gfs_ds['tp'] - gfs_datasets[-1]['tp']
    gfs_datasets.append(gfs_ds)

    if ecmwf:
        for var in ecmwf_ds.data_vars:
            ecmwf_ds[var] = ecmwf_ds[var].astype('float32')

    if ecmwf:
        new_ecmwf_ds = ecmwf_ds.copy()
        if frame>3:
            new_ecmwf_ds['tp'] = ecmwf_ds['tp'] - raw_ecmwf_datasets[-1]['tp']
        raw_ecmwf_datasets.append(ecmwf_ds)
        ecmwf_datasets.append(new_ecmwf_ds)


gfs_ds = xr.concat(gfs_datasets, dim='time')
gfs_tp_arr = np.expand_dims(gfs_ds.tp.values, axis=1)

if not ecmwf:
    ecmwf_datasets = gfs_datasets
ecmwf_ds = xr.concat(ecmwf_datasets, dim='time')
if ecmwf:
    ecmwf_ds['tp'] = ecmwf_ds['tp']*1000
ecmwf_tp_arr = np.expand_dims(ecmwf_ds.tp.values, axis=1)

gfs_arr = torch.from_numpy(gfs_tp_arr).to('mps')
ecmwf_arr = torch.from_numpy(ecmwf_tp_arr).to('mps')

gfs_preds = []
ecmwf_preds = []

def predict(epoch, gfs_arr, ecmwf_arr):
    model = UNetWithAttention(1, 1).to('mps')
    model.load_state_dict(torch.load(f'/Users/clamalo/documents/harpnet/checkpoints/train_e{epoch}.pt')['model_state_dict'])
    model.eval()
    with torch.no_grad():
        gfs_pred = model(gfs_arr).squeeze()
        ecmwf_pred = model(ecmwf_arr).squeeze()
    return gfs_pred, ecmwf_pred


first_member = 20
last_member = 21
for epoch in tqdm(range(first_member, last_member)):
    gfs_pred, ecmwf_pred = predict(epoch, gfs_arr, ecmwf_arr)
    gfs_preds.append(gfs_pred)
    ecmwf_preds.append(ecmwf_pred)
    torch.cuda.empty_cache()  
gfs_pred = torch.stack(gfs_preds)
ecmwf_pred = torch.stack(ecmwf_preds)


gfs_output_ds = xr.Dataset(
    {'tp': (['member', 'time', 'latitude', 'longitude'], gfs_pred.cpu().detach().numpy())},
    coords={
        'member': np.arange(gfs_pred.shape[0]),
        'time': gfs_ds.valid_time,
        'latitude': ds_latitudes,
        'longitude': ds_longitudes
    }
)
ecmwf_output_ds = xr.Dataset(
    {'tp': (['member', 'time', 'latitude', 'longitude'], ecmwf_pred.cpu().detach().numpy())},
    coords={
        'member': np.arange(ecmwf_pred.shape[0]),
        'time': ecmwf_ds.valid_time,
        'latitude': ds_latitudes,
        'longitude': ds_longitudes
    }
)

mean_gfs_ds = gfs_output_ds.mean(dim='member')
mean_ecmwf_ds = ecmwf_output_ds.mean(dim='member')

gfs_input_ds_sum = gfs_ds.sum(dim='time')
gfs_pred_ds_sum = mean_gfs_ds.sum(dim='time')
ecmwf_input_ds_sum = ecmwf_ds.sum(dim='time')
ecmwf_pred_ds_sum = mean_ecmwf_ds.sum(dim='time')

import matplotlib.colors as colors
def weatherbell_precip_colormap():
    cmap = colors.ListedColormap(['#ffffff', '#bdbfbd','#aba5a5', '#838383', '#6e6e6e', '#b5fbab', '#95f58b','#78f572', '#50f150','#1eb51e', '#0ea10e', '#1464d3', '#2883f1', '#50a5f5','#97d3fb', '#b5f1fb','#fffbab', '#ffe978', '#ffc13c', '#ffa100', '#ff6000','#ff3200', '#e11400','#c10000', '#a50000', '#870000', '#643c31', '#8d6558','#b58d83', '#c7a095','#f1ddd3', '#cecbdc'])#, '#aca0c7', '#9b89bd', '#725ca3','#695294', '#770077','#8d008d', '#b200b2', '#c400c4', '#db00db'])
    return cmap
colormap = weatherbell_precip_colormap()
colormap.set_over(color='#aca0c7')
bounds = [0,0.01,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.5,3,3.5,4,5,6,7,8,9,10]#,11,12,13,14,15,16,17,18,19,20])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))


#plot the sums all together with the colormap and norm and bounds above
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
for ax in axs.flat:
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES, linestyle=':')
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', linewidth=0.5)
min_latitude, max_latitude, min_longitude, max_longitude = ds_latitudes[0], ds_latitudes[-1], ds_longitudes[0], ds_longitudes[-1]

axs[0, 0].set_extent([min_longitude, max_longitude, min_latitude, max_latitude], crs=ccrs.PlateCarree())
# mesh0 = axs[0, 0].pcolormesh(gfs_input_ds_sum['longitude'], gfs_input_ds_sum['latitude'], gfs_input_ds_sum['tp']*0.0393701, transform=ccrs.PlateCarree(), cmap=colormap, norm=norm)
mesh0 = axs[0, 0].contourf(gfs_input_ds_sum['longitude'], gfs_input_ds_sum['latitude'], gfs_input_ds_sum['tp']*0.0393701, transform=ccrs.PlateCarree(), levels=bounds, cmap=colormap, norm=norm)
# fig.colorbar(mesh0, ax=axs[0, 0], orientation='horizontal', label='Value')
axs[0, 0].set_title('0.25 Deg Input (Filled Contour)')

axs[1, 0].set_extent([min_longitude, max_longitude, min_latitude, max_latitude], crs=ccrs.PlateCarree())
mesh0 = axs[1, 0].pcolormesh(gfs_input_ds_sum['longitude'], gfs_input_ds_sum['latitude'], gfs_input_ds_sum['tp']*0.0393701, transform=ccrs.PlateCarree(), cmap=colormap, norm=norm)
# mesh0 = axs[1, 0].contourf(ecmwf_input_ds_sum['longitude'], ecmwf_input_ds_sum['latitude'], ecmwf_input_ds_sum['tp']*0.0393701, transform=ccrs.PlateCarree(), levels=bounds, cmap=colormap, norm=norm)
# fig.colorbar(mesh0, ax=axs[1, 0], orientation='horizontal', label='Value')
axs[1, 0].set_title('0.25 Deg Input (Gridded Mesh)')

axs[0, 1].set_extent([min_longitude, max_longitude, min_latitude, max_latitude], crs=ccrs.PlateCarree())
# mesh1 = axs[0, 1].pcolormesh(gfs_pred_ds_sum['longitude'], gfs_pred_ds_sum['latitude'], gfs_pred_ds_sum['tp'].values*0.0393701, transform=ccrs.PlateCarree(), cmap=colormap, norm=norm)
mesh1 = axs[0, 1].contourf(gfs_pred_ds_sum['longitude'], gfs_pred_ds_sum['latitude'], gfs_pred_ds_sum['tp'].values*0.0393701, transform=ccrs.PlateCarree(), levels=bounds, cmap=colormap, norm=norm)
# fig.colorbar(mesh1, ax=axs[0, 1], orientation='horizontal', label='Value')
axs[0, 1].set_title('8km Downscaled Output (Filled Contour)')

axs[1, 1].set_extent([min_longitude, max_longitude, min_latitude, max_latitude], crs=ccrs.PlateCarree())
mesh1 = axs[1, 1].pcolormesh(gfs_pred_ds_sum['longitude'], gfs_pred_ds_sum['latitude'], gfs_pred_ds_sum['tp'].values*0.0393701, transform=ccrs.PlateCarree(), cmap=colormap, norm=norm)
# mesh1 = axs[1, 1].contou rf(ecmwf_pred_ds_sum['longitude'], ecmwf_pred_ds_sum['latitude'], ecmwf_pred_ds_sum['tp'].values*0.0393701, transform=ccrs.PlateCarree(), levels=bounds, cmap=colormap, norm=norm)
# fig.colorbar(mesh1, ax=axs[1, 1], orientation='horizontal', label='Value')
axs[1, 1].set_title('8km Downscaled Output (Gridded Mesh)')

plt.savefig('comp.png')


coordinates = [40.64835, -106.68377]
gfs_output_arr = gfs_output_ds['tp'].sel(latitude=coordinates[0], longitude=coordinates[1], method='nearest').values
gfs_input_arr = gfs_ds['tp'].sel(latitude=coordinates[0], longitude=coordinates[1], method='nearest').values
print(f'GFS Input: {np.sum(gfs_input_arr)*0.0393701}')
print(f'GFS Output: {np.sum(np.mean(gfs_output_arr, axis=0))*0.0393701}')

#current shape is (member, time, lat, lon) of 3hourly precip. i want to plot a time-series of each member *accumulated* precip over time
gfs_output_arr = np.cumsum(gfs_output_arr, axis=1)
gfs_output_arr = gfs_output_arr*0.0393701
fig, ax = plt.subplots()

# for member in range(num_members):
#     alpha_value = (member + 1) / num_members  # Calculate alpha value
#     ax.plot(gfs_ds.valid_time, gfs_output_arr[member], label=f'Member {member}', linestyle='-', color='blue', alpha=alpha_value)

for member in range(gfs_output_arr.shape[0]):
    ax.plot(gfs_ds.valid_time, gfs_output_arr[member], label=f'Member {member}', linestyle='-', color='blue', alpha=0.6)

ax.plot(gfs_ds.valid_time, np.cumsum(gfs_input_arr)*0.0393701, label='Input', linestyle='--', color='black')
ax.plot(gfs_ds.valid_time, np.mean(gfs_output_arr, axis=0), label='Mean Output', linestyle='--', color='purple', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('Accumulated Precip (in)')
ax.set_title('GFS Output')
# ax.legend()
ax.tick_params(axis='x', labelsize='small', rotation=45)
plt.savefig('gfs_output_ts.png')
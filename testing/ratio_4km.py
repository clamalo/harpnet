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
sys.path.append(os.path.abspath(os.path.join('utils')))
from utils import *
from model import UNetWithAttention
import matplotlib.colors as colors
from datetime import datetime, timedelta

def weatherbell_precip_colormap():
    cmap = colors.ListedColormap(['#ffffff', '#bdbfbd','#aba5a5', '#838383', '#6e6e6e', '#b5fbab', '#95f58b','#78f572', '#50f150','#1eb51e', '#0ea10e', '#1464d3', '#2883f1', '#50a5f5','#97d3fb', '#b5f1fb','#fffbab', '#ffe978', '#ffc13c', '#ffa100', '#ff6000','#ff3200', '#e11400','#c10000', '#a50000', '#870000', '#643c31', '#8d6558','#b58d83', '#c7a095','#f1ddd3', '#cecbdc'])#, '#aca0c7', '#9b89bd', '#725ca3','#695294', '#770077','#8d008d', '#b200b2', '#c400c4', '#db00db'])
    return cmap
colormap = weatherbell_precip_colormap()
colormap.set_over(color='#aca0c7')
bounds = [0,0.01,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.5,3,3.5,4,5,6,7,8,9,10]#,11,12,13,14,15,16,17,18,19,20])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))

datestr = '20211220'
cycle = '00'
datestr_cycle_datetime = datetime.strptime(datestr + cycle, '%Y%m%d%H')
max_lat, min_lat, max_lon, min_lon = 44.85, 39.3, -108.7, -114.25


with open(os.path.join('/Users/clamalo/documents/harpnet/load_data/shapes.pkl'), 'rb') as f:
    shapes = pickle.load(f)
ds_latitudes = shapes['output_latitudes']
ds_longitudes = shapes['output_longitudes']

reference_ds = xr.open_dataset(f'/Users/clamalo/documents/harpnet/load_data/reference_ds.grib2')
reference_ds = reference_ds.assign_coords(longitude=(((reference_ds.longitude + 180) % 360) - 180)).sortby('longitude')
reference_ds = reference_ds.sel(latitude=slice(max_lat+0.5, min_lat-0.5), longitude=slice(min_lon-0.5, max_lon+0.5))
print(reference_ds)

gfs_datasets = []
for frame in range(3,385,3):
    time_to_sel = datestr_cycle_datetime + timedelta(hours=frame)

    year, month = time_to_sel.year, time_to_sel.month

    monthly_cropped_ds = xr.open_dataset(f'/Volumes/T9/monthly/{year}-{month:02}.nc')
    monthly_cropped_ds = monthly_cropped_ds.sel(time=time_to_sel)
    
    ds = monthly_cropped_ds.interp(lat=reference_ds.latitude, lon=reference_ds.longitude)

    gfs_datasets.append(ds)

gfs_ds = xr.concat(gfs_datasets, dim='time')
gfs_arr = np.expand_dims(gfs_ds.tp.values, axis=1)
gfs_arr = gfs_arr.astype(np.float32)
gfs_arr = torch.from_numpy(gfs_arr).to('mps')


gfs_preds = []
def predict(epoch, gfs_arr):
    model = UNetWithAttention(1, 1).to('mps')
    model.load_state_dict(torch.load(f'/Users/clamalo/documents/harpnet/checkpoints/train_e{epoch}.pt')['model_state_dict'])
    model.eval()
    with torch.no_grad():
        gfs_pred = model(gfs_arr).squeeze()
    return gfs_pred


# first_member = 60
# last_member = 61
# members_to_process = range(first_member, last_member)
members_to_process = [82, 70, 83, 69, 25]
for epoch in tqdm(members_to_process):
    gfs_pred = predict(epoch, gfs_arr)
    gfs_pred = torch.where(gfs_pred < 0, 0, gfs_pred) 
    gfs_preds.append(gfs_pred)
    torch.cuda.empty_cache()  
gfs_pred = torch.stack(gfs_preds)



gfs_output_ds = xr.Dataset(
    {'tp': (['member', 'time', 'latitude', 'longitude'], gfs_pred.cpu().detach().numpy())},
    coords={
        'member': np.arange(gfs_pred.shape[0]),
        'time': gfs_ds.time.values,
        'latitude': ds_latitudes,
        'longitude': ds_longitudes
    }
)

mean_gfs_ds = gfs_output_ds.mean(dim='member')
#make all values less than 0.001 equal to 0
mean_gfs_ds['tp'] = xr.where((mean_gfs_ds['tp']*0.0393701) < 0.001, 0, mean_gfs_ds['tp']) 

for t in range(len(mean_gfs_ds.time)): 
    continue
    # Extract and format the valid time
    valid_time = mean_gfs_ds.time.values[t]
    valid_time_str = np.datetime_as_string(valid_time, unit='h')
    
    # Plot two subplots: left is the input, right is the output
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    for ax in axs.flat:
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES, linestyle=':')
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', linewidth=0.5)
    
    min_latitude, max_latitude, min_longitude, max_longitude = ds_latitudes[0], ds_latitudes[-1], ds_longitudes[0], ds_longitudes[-1]

    axs[0].set_extent([min_longitude, max_longitude, min_latitude, max_latitude], crs=ccrs.PlateCarree())
    mesh0 = axs[0].pcolormesh(gfs_ds['longitude'], gfs_ds['latitude'], gfs_ds.tp.isel(time=t).values*0.0393701, transform=ccrs.PlateCarree(), vmin=0, vmax=0.4)#, norm=norm, cmap=colormap)
    # mesh0 = axs[0].contourf(gfs_ds['longitude'], gfs_ds['latitude'], gfs_ds.tp.isel(time=t).values*0.0393701, transform=ccrs.PlateCarree())#, levels=bounds, cmap=colormap, norm=norm)
    fig.colorbar(mesh0, ax=axs[0], orientation='horizontal', label='Value')
    axs[0].set_title(f'Input - Valid Time: {valid_time_str}')

    axs[1].set_extent([min_longitude, max_longitude, min_latitude, max_latitude], crs=ccrs.PlateCarree())
    mesh1 = axs[1].pcolormesh(mean_gfs_ds['longitude'], mean_gfs_ds['latitude'], mean_gfs_ds.tp.isel(time=t).values*0.0393701, transform=ccrs.PlateCarree(), vmin=0, vmax=0.4)#, norm=norm, cmap=colormap)
    # mesh1 = axs[1].contourf(mean_gfs_ds['longitude'], mean_gfs_ds['latitude'], mean_gfs_ds.tp.isel(time=t).values*0.0393701, transform=ccrs.PlateCarree())#, levels=bounds, cmap=colormap, norm=norm)
    fig.colorbar(mesh1, ax=axs[1], orientation='horizontal', label='Value')
    axs[1].set_title(f'Output - Valid Time: {valid_time_str}')
    
    plt.savefig(f'output_{t}.png')
    plt.close(fig)  # Close the figure to free up memory









gfs_input_ds_sum = gfs_ds.sum(dim='time')
gfs_pred_ds_sum = mean_gfs_ds.sum(dim='time') 


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
axs[0, 1].set_title('4km Downscaled Output (Filled Contour)')

axs[1, 1].set_extent([min_longitude, max_longitude, min_latitude, max_latitude], crs=ccrs.PlateCarree())
mesh1 = axs[1, 1].pcolormesh(gfs_pred_ds_sum['longitude'], gfs_pred_ds_sum['latitude'], gfs_pred_ds_sum['tp'].values*0.0393701, transform=ccrs.PlateCarree(), cmap=colormap, norm=norm)
# mesh1 = axs[1, 1].contou rf(ecmwf_pred_ds_sum['longitude'], ecmwf_pred_ds_sum['latitude'], ecmwf_pred_ds_sum['tp'].values*0.0393701, transform=ccrs.PlateCarree(), levels=bounds, cmap=colormap, norm=norm)
# fig.colorbar(mesh1, ax=axs[1, 1], orientation='horizontal', label='Value')
axs[1, 1].set_title('4km Downscaled Output (Gridded Mesh)')

plt.savefig('comp.png')



gfs_input_ds_sum = gfs_input_ds_sum.interp(latitude=gfs_pred_ds_sum['latitude'], longitude=gfs_pred_ds_sum['longitude'])
ratio_ds = gfs_pred_ds_sum/gfs_input_ds_sum

#plot the ratio
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES, linestyle=':')
ax.add_feature(cfeature.BORDERS)
ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', linewidth=0.5)
ax.set_extent([min_longitude, max_longitude, min_latitude, max_latitude], crs=ccrs.PlateCarree())
mesh = ax.pcolormesh(ratio_ds['longitude'], ratio_ds['latitude'], ratio_ds['tp'].values, transform=ccrs.PlateCarree(), vmin=0, vmax=2, cmap='coolwarm')
fig.colorbar(mesh, ax=ax, orientation='horizontal', label='Value')
ax.set_title('4km Downscaled Output / 0.25 Deg Input')
plt.savefig('ratio.png')





stations = {
    'Snowbird': [40.57, -111.66],
    'Powder Mountain': [41.36663, -111.76666],
    'Grand Targhee': [43.79000, -110.93000],
    'Thaynes Canyon': [40.62998, -111.52994],
}

for location, coordinates in stations.items():

    gfs_output_arr = gfs_output_ds['tp'].sel(latitude=coordinates[0], longitude=coordinates[1], method='nearest').values
    gfs_input_arr = gfs_ds['tp'].sel(latitude=coordinates[0], longitude=coordinates[1], method='nearest').values
    print(f'{location} Input: {np.sum(gfs_input_arr)*0.0393701}')
    print(f'{location} Output: {np.sum(np.mean(gfs_output_arr, axis=0))*0.0393701}')

    #current shape is (member, time, lat, lon) of 3hourly precip. i want to plot a time-series of each member *accumulated* precip over time
    gfs_output_arr = np.cumsum(gfs_output_arr, axis=1)
    gfs_output_arr = gfs_output_arr*0.0393701
    fig, ax = plt.subplots()

    for member in range(gfs_output_arr.shape[0]):
        ax.plot(gfs_ds.time.values, gfs_output_arr[member], label=f'Member {member}', linestyle='-', color='blue', alpha=0.6)

    ax.plot(gfs_ds.time.values, np.cumsum(gfs_input_arr)*0.0393701, label='Input', linestyle='--', color='black')
    ax.plot(gfs_ds.time.values, np.mean(gfs_output_arr, axis=0), label='Mean Output', linestyle='--', color='purple', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Accumulated Precip (in)')
    ax.set_title(f'{location} Output')
    # ax.legend()
    ax.tick_params(axis='x', labelsize='small', rotation=45)
    plt.savefig(f'{location}_gfs_output_ts.png')
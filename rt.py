import os
import numpy as np
import xarray as xr
import pandas as pd
import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
import sys
from metpy.plots import USCOUNTIES
import requests
import glob
from datetime import datetime, timedelta

cwd_dir = (os.path.abspath(os.path.join(os.getcwd())))
sys.path.insert(0, cwd_dir)
from utils.utils import *
from utils.model import UNetWithAttention
import utils.constants as constants


# Constants
pad = True
realtime = True
rt_model = 'ecmwf'
ingest = True
datestr, cycle = '20240827', '12'
frames = range(3, 13, 3)
# sort_epochs([0])




setup()

# Create stitch figure directory if it doesn't exist
fig_dir = constants.figures_dir + 'stitch'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

num_times = len(frames)


if realtime and ingest:
    files = glob.glob('utils/data/*')
    for f in files:
        os.remove(f)
    print("Downloading rt data")
    if rt_model == 'gfs':
        for frame in tqdm(frames):
            ingest_gfs_data(datestr, cycle, frame)
    elif rt_model == 'ecmwf':
        ingest_ecmwf_data(datestr, cycle, list(frames))
    elif rt_model == 'eps':
        req_param = ["tp"]
        for frame in tqdm(frames):
            asyncio.run(ingest_eps_data(datestr, cycle, frame, req_param, None, f"utils/data/eps_{frame}.grib2", mode='wb'))


# Load grid domains
create_grid_domains()
with open(f'{constants.domains_dir}grid_domains.pkl', 'rb') as f:
    grid_domains = pickle.load(f)
    num_domains = len(grid_domains)


if not realtime:
    desired_start_time, desired_end_time = datetime.strptime(datestr+cycle, '%Y%m%d%H')+timedelta(hours=frames[0]), datetime.strptime(datestr+cycle, '%Y%m%d%H')+timedelta(hours=frames[-1])

    year_month_pairs = [(y, m) for y in range(desired_start_time.year, desired_end_time.year + 1)
          for m in range(1 if y > desired_start_time.year else desired_start_time.month,
                         13 if y < desired_end_time.year else desired_end_time.month + 1)]

    datasets = []
    for year, month in year_month_pairs:
        ds = xr.open_dataset(f'{constants.nc_dir}{year}-{month:02d}.nc')
        time_index = pd.DatetimeIndex(ds.time.values)
        filtered_times = time_index[time_index.hour.isin([3, 6, 9, 12, 15, 18, 21, 0])]
        ds = ds.sel(time=filtered_times).sortby('time')
        datasets.append(ds)
    master_ds = xr.concat(datasets, dim='time')
    desired_start_time, desired_end_time = datetime.strptime(datestr+cycle, '%Y%m%d%H')+timedelta(hours=frames[0]), datetime.strptime(datestr+cycle, '%Y%m%d%H')+timedelta(hours=frames[-1])
    desired_start_time_index, desired_end_time_index = np.where(master_ds.time.values == np.datetime64(desired_start_time)), np.where(master_ds.time.values == np.datetime64(desired_end_time))
    if len(desired_start_time_index[0]) == 0 or len(desired_end_time_index[0]) == 0:
        raise ValueError('Desired start or end time is not in the master dataset')
    master_ds = master_ds.sel(time=master_ds.time[desired_start_time_index[0][0]:desired_end_time_index[0][0]+1])


# Initialize master fine latitude and longitude sets
available_domains = range(0, num_domains)
master_fine_lats, master_fine_lons = set(), set()
master_coarse_lats, master_coarse_lons = set(), set()
for domain in available_domains:
    fine_lats, fine_lons, coarse_lats, coarse_lons = get_lats_lons(domain, pad=True)
    master_fine_lats.update(fine_lats)
    master_fine_lons.update(fine_lons)
    master_coarse_lats.update(coarse_lats)
    master_coarse_lons.update(coarse_lons)
master_fine_lats = np.sort(np.array(list(master_fine_lats)))
master_fine_lons = np.sort(np.array(list(master_fine_lons)))
master_coarse_lats = np.sort(np.array(list(master_coarse_lats)))
master_coarse_lons = np.sort(np.array(list(master_coarse_lons)))


# Initialize fake data array
fine_arr = np.zeros((num_times, len(master_fine_lats), len(master_fine_lons)))
coarse_arr = np.zeros((num_times, len(master_coarse_lats), len(master_coarse_lons)))


# Load reference dataset
reference_ds = xr.load_dataset(f'utils/reference_ds.grib2', engine='cfgrib')
reference_ds = reference_ds.assign_coords(longitude=(((reference_ds.longitude + 180) % 360) - 180)).sortby('longitude')
reference_ds = reference_ds.sortby('latitude', ascending=True)


# Load model
model = UNetWithAttention(1, 1, output_shape=(64, 64)).to(constants.device)

# Define initial indices for the fine and coarse grids
fine_lat_idx, fine_lon_idx = 0, 0
coarse_lat_idx, coarse_lon_idx = 0, 0
fine_lat_step, fine_lon_step = 64, 64
coarse_lat_step, coarse_lon_step = 16, 16

test_losses = []
bilinear_losses = []

# Iterate through each available domain
for domain in tqdm(available_domains):
    min_lat, max_lat, min_lon, max_lon = grid_domains[domain]

    # Crop reference dataset based on whether padding is applied
    if pad:
        cropped_reference_ds = reference_ds.sel(
            latitude=slice(min_lat - 0.25, max_lat),
            longitude=slice(min_lon - 0.25, max_lon)
        )
    else:
        cropped_reference_ds = reference_ds.sel(
            latitude=slice(min_lat, max_lat - 0.25),
            longitude=slice(min_lon, max_lon - 0.25)
        )

    # Extract latitudes and longitudes from the cropped reference dataset
    cropped_lats = cropped_reference_ds.latitude.values
    cropped_lons = cropped_reference_ds.longitude.values

    if realtime:
        if rt_model == 'gfs':
            master_ds = xr.open_mfdataset(f'utils/data/gfs*.grib', engine='cfgrib', combine='nested', concat_dim='valid_time')
            master_ds = master_ds.sortby('valid_time')
            for t in range(len(master_ds.valid_time)):
                if master_ds['valid_time'][t].dt.hour in [0, 6, 12, 18]:
                    master_ds['tp'][t] = master_ds['tp'][t]-master_ds['tp'][t-1]
            master_ds = master_ds.rename({'latitude': 'lat', 'longitude': 'lon'})
            master_ds = master_ds.sortby('lat', ascending=True)
            master_ds = master_ds.assign_coords(lon=(((master_ds.lon + 180) % 360) - 180)).sortby('lon')
            times = master_ds.valid_time.values
        elif rt_model == 'ecmwf':
            master_ds = xr.open_dataset(f'utils/data/ecmwf.grib', engine='cfgrib')
            master_ds = master_ds.sortby('valid_time')
            master_ds['cum_tp'] = master_ds['tp']*1000
            for t in range(1,len(master_ds.valid_time)):
                master_ds['tp'][t] = master_ds['cum_tp'][t]-master_ds['cum_tp'][t-1]
            master_ds = master_ds.rename({'latitude': 'lat', 'longitude': 'lon'})
            master_ds = master_ds.sortby('lat', ascending=True)
            master_ds = master_ds.assign_coords(lon=(((master_ds.lon + 180) % 360) - 180)).sortby('lon')
            times = master_ds.valid_time.values
    else:
        times = master_ds.time.values

    # Interpolate the master dataset to match the reference dataset's grid
    coarse_ds = master_ds.interp(lat=cropped_lats, lon=cropped_lons)

    # Load checkpoint and evaluate the model if the checkpoint exists
    checkpoint_path = f'{constants.checkpoints_dir}best/{domain}_model.pt'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        # Calculate and print the percentage error reduction
        percent_error_reduction = (1 - (np.sqrt(checkpoint['test_loss']) / np.sqrt(checkpoint['bilinear_loss']))) * 100
        print(f'{domain}: {percent_error_reduction:.2f}%')
        test_losses.append(checkpoint['test_loss'])
        bilinear_losses.append(checkpoint['bilinear_loss'])

        # Load model state and set to evaluation mode
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Process the coarse dataset with the model
        tp = torch.tensor(coarse_ds.tp.values, dtype=torch.float32).to(constants.device)
        output = model(tp).cpu().detach().numpy() * 0.0393701
        fine_arr[:, fine_lat_idx:fine_lat_idx+fine_lat_step, fine_lon_idx:fine_lon_idx+fine_lon_step] = output
        coarse_arr[:, coarse_lat_idx:coarse_lat_idx+coarse_lat_step, coarse_lon_idx:coarse_lon_idx+coarse_lon_step] = coarse_ds.tp.values[:, 1:-1, 1:-1] * 0.0393701
    else:
        fine_arr[:, fine_lat_idx:fine_lat_idx+fine_lat_step, fine_lon_idx:fine_lon_idx+fine_lon_step] = np.nan
        coarse_arr[:, coarse_lat_idx:coarse_lat_idx+coarse_lat_step, coarse_lon_idx:coarse_lon_idx+coarse_lon_step] = np.nan

    # Increment longitude indices for the next domain
    fine_lon_idx += fine_lon_step
    coarse_lon_idx += coarse_lon_step

    # If the end of the longitude is reached, reset longitude indices and increment latitude indices
    if fine_lon_idx >= len(master_fine_lons):
        fine_lon_idx = 0
        coarse_lon_idx = 0
        fine_lat_idx += fine_lat_step
        coarse_lat_idx += coarse_lat_step

fine_arr_sum = np.sum(fine_arr, axis=0)
coarse_arr_sum = np.sum(coarse_arr, axis=0)

# Determine the extent based on non-NaN data
non_nan_indices = np.where(~np.isnan(fine_arr_sum))
# Get the corresponding latitude and longitude bounds
min_lat_idx, max_lat_idx = non_nan_indices[0].min(), non_nan_indices[0].max()
min_lon_idx, max_lon_idx = non_nan_indices[1].min(), non_nan_indices[1].max()
min_lat, max_lat = master_fine_lats[min_lat_idx], master_fine_lats[max_lat_idx]
min_lon, max_lon = master_fine_lons[min_lon_idx], master_fine_lons[max_lon_idx]

colormap, norm, bounds = weatherbell_precip_colormap()

# Plot each time step with the calculated extent
for i in tqdm(range(num_times)):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    ax.add_feature(cfeature.STATES.with_scale('10m'))
    # ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', alpha=0.75, linewidth=0.5)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    cf = ax.pcolormesh(master_fine_lons, master_fine_lats, fine_arr[i], transform=ccrs.PlateCarree(), vmin=0, vmax=0.5)#, cmap=colormap, norm=norm)
    # cf = ax.contourf(master_fine_lons, master_fine_lats, fine_arr[i], transform=ccrs.PlateCarree(), levels=bounds, cmap=colormap, norm=norm)
    plt.colorbar(cf, ax=ax, orientation='horizontal', label='tp (in)', pad=0.02)
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f'HARPNET Output {times[i]}', y=0.95)  # Increase the y parameter to move the title up
    plt.savefig(f'{fig_dir}/{i}.png', bbox_inches='tight')
    plt.close()

# Plot the sum with the calculated extent
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.coastlines()
ax.add_feature(cfeature.STATES.with_scale('10m'))
# ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', alpha=0.75, linewidth=0.5)
ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
cf = ax.pcolormesh(master_fine_lons, master_fine_lats, fine_arr_sum, transform=ccrs.PlateCarree())#, cmap=colormap, norm=norm)
# cf = ax.contourf(master_fine_lons, master_fine_lats, fine_arr_sum, transform=ccrs.PlateCarree(), levels=bounds, cmap=colormap, norm=norm)
plt.colorbar(cf, ax=ax, orientation='horizontal', label='tp (in)', pad=0.02)
plt.subplots_adjust(top=0.9)
plt.suptitle(f'HARPNET Summed Output: {times[0]}-{times[-1]}')
plt.savefig(f'{fig_dir}/sum.png', bbox_inches='tight')
plt.close()


#now plot the fine and coarse sum arrays side by side
fig, ax = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax[0].coastlines()
ax[0].add_feature(cfeature.STATES.with_scale('10m'))
# ax[0].add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', alpha=0.75, linewidth=0.5)
ax[0].set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
vmin = 0
vmax = np.nanmax(fine_arr_sum)
# cf = ax[0].pcolormesh(master_fine_lons, master_fine_lats, fine_arr_sum, transform=ccrs.PlateCarree())#, cmap=colormap, norm=norm)
cf = ax[0].pcolormesh(master_fine_lons, master_fine_lats, fine_arr_sum, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
# cf = ax[0].contourf(master_fine_lons, master_fine_lats, fine_arr_sum, transform=ccrs.PlateCarree(), levels=bounds, cmap=colormap, norm=norm)
plt.colorbar(cf, ax=ax[0], orientation='horizontal', label='tp (in)', pad=0.02)
ax[0].set_title('Fine Sum')

ax[1].coastlines()
ax[1].add_feature(cfeature.STATES.with_scale('10m'))
# ax[1].add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', alpha=0.75, linewidth=0.5)
ax[1].set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
# cf = ax[1].pcolormesh(master_coarse_lons, master_coarse_lats, coarse_arr_sum, transform=ccrs.PlateCarree())#, cmap=colormap, norm=norm)
cf = ax[1].pcolormesh(master_coarse_lons, master_coarse_lats, coarse_arr_sum, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
# cf = ax[1].contourf(master_coarse_lons, master_coarse_lats, coarse_arr_sum, transform=ccrs.PlateCarree(), levels=bounds, cmap=colormap, norm=norm)
plt.colorbar(cf, ax=ax[1], orientation='horizontal', label='tp (in)', pad=0.02)
ax[1].set_title('Coarse Sum')

plt.subplots_adjust(top=0.9)
plt.suptitle(f'HARPNET Summed Output: {times[0]}-{times[-1]}')
plt.savefig(f'{fig_dir}/sum_side_by_side.png', bbox_inches='tight')
plt.close()


ds = xr.Dataset(
    {
        'tp': (['time', 'lat', 'lon'], fine_arr)
    },
    coords={
        'time': times,
        'lat': master_fine_lats,
        'lon': master_fine_lons
    }
)

coarse_ds = xr.Dataset(
    {
        'tp': (['time', 'lat', 'lon'], coarse_arr)
    },
    coords={
        'time': times,
        'lat': master_coarse_lats,
        'lon': master_coarse_lons
    }
)

# ds cumsum
ds['tp'] = ds.tp.cumsum(dim='time')
coarse_ds['tp'] = coarse_ds.tp.cumsum(dim='time')
coarse_ds = coarse_ds.interp(lat=ds.lat, lon=ds.lon)

points = {
    'Echo Peak': (38.85, -120.08),
    'Palisades Tahoe': (39.19, -120.27),
    'Mt. Rose': (39.32, -119.89)
}

# plot the cumulative precipitation at each point from ds as a time series
fig, ax = plt.subplots(figsize=(10, 5))
for point in points:
    ds.tp.sel(lat=points[point][0], lon=points[point][1], method='nearest').plot(ax=ax, label=point)
ax.set_xlabel('Time')
ax.set_ylabel('tp (in)')
ax.set_title('Cumulative Precipitation at Various Points')
ax.legend()
plt.savefig(f'{fig_dir}/points_cumsum.png', bbox_inches='tight')
plt.close()

for point in points:
    print(f'{point}: {ds.tp.sel(lat=points[point][0], lon=points[point][1], method="nearest").values[-1]}", {coarse_ds.tp.sel(lat=points[point][0], lon=points[point][1], method="nearest").values[-1]}"')
    
print('Average Test Loss:', np.mean(test_losses))
print('Average Bilinear Loss:', np.mean(bilinear_losses))
print('Average Percent Error Reduction:', (1 - (np.sqrt(np.mean(test_losses)) / np.sqrt(np.mean(bilinear_losses)))) * 100)
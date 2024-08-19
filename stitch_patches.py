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
from utils.utils import *
from utils.model import UNetWithAttention
from metpy.plots import USCOUNTIES
# sort_epochs()

# Create stitch figure directory if it doesn't exist
fig_dir = 'figures/stitch'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# Constants
device = 'mps'
pad = True


def weatherbell_precip_colormap():
    cmap = colors.ListedColormap(['#ffffff', '#bdbfbd','#aba5a5', '#838383', '#6e6e6e', '#b5fbab', '#95f58b','#78f572', '#50f150','#1eb51e', '#0ea10e', '#1464d3', '#2883f1', '#50a5f5','#97d3fb', '#b5f1fb','#fffbab', '#ffe978', '#ffc13c', '#ffa100', '#ff6000','#ff3200', '#e11400','#c10000', '#a50000', '#870000', '#643c31', '#8d6558','#b58d83', '#c7a095','#f1ddd3', '#cecbdc'])#, '#aca0c7', '#9b89bd', '#725ca3','#695294', '#770077','#8d008d', '#b200b2', '#c400c4', '#db00db'])
    return cmap
colormap = weatherbell_precip_colormap()
colormap.set_over(color='#aca0c7')
bounds = [0,0.01,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.5,3,3.5,4,5,6,7,8,9,10]#,11,12,13,14,15,16,17,18,19,20])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))


# Load and filter master dataset
master_ds = xr.open_dataset(f'{constants.nc_dir}{2017}-{2:02d}.nc')
time_index = pd.DatetimeIndex(master_ds.time.values)
filtered_times = time_index[time_index.hour.isin([3, 6, 9, 12, 15, 18, 21, 0])]
master_ds = master_ds.sel(time=filtered_times).sortby('time')
# master_ds = master_ds.sel(time=master_ds.time[0:224])
master_ds = master_ds.sel(time=master_ds.time[0:100])
times = master_ds.time.values
num_times = len(times)


# Initialize master fine latitude and longitude sets
available_domains = range(0, 49)
master_fine_lats, master_fine_lons = set(), set()
master_coarse_lats, master_coarse_lons = set(), set()
for domain in available_domains:
    fine_lats, fine_lons, coarse_lats, coarse_lons = get_lats_lons(domain, pad=False)
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
reference_ds = xr.load_dataset(f'{constants.base_dir}reference_ds.grib2', engine='cfgrib')
reference_ds = reference_ds.assign_coords(longitude=(((reference_ds.longitude + 180) % 360) - 180)).sortby('longitude')
reference_ds = reference_ds.sortby('latitude', ascending=True)


# Load grid domains
with open(f'{constants.domains_dir}grid_domains.pkl', 'rb') as f:
    grid_domains = pickle.load(f)

# Load model
model = UNetWithAttention(1, 1, output_shape=(64, 64)).to(device)

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
        tp = torch.tensor(coarse_ds.tp.values, dtype=torch.float32).to(device)
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
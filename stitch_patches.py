import os
import numpy as np
import xarray as xr
import pandas as pd
import pickle
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
from utils.utils import *
from utils.model import UNetWithAttention
from metpy.plots import USCOUNTIES
from sort_epochs import sort_epochs
sort_epochs()

# Create stitch figure directory if it doesn't exist
fig_dir = 'figures/stitch'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)


# Constants
start_time, end_time = 48, 185
num_times = end_time - start_time
year, month = 2017, 2
nc_file = f'{constants.nc_dir}{year}-{month:02d}.nc'
reference_file = f'{constants.base_dir}reference_ds.grib2'
grid_domains_file = f'{constants.domains_dir}grid_domains.pkl'
device = 'mps'
pad = True


import matplotlib.colors as colors
def weatherbell_precip_colormap():
    cmap = colors.ListedColormap(['#ffffff', '#bdbfbd','#aba5a5', '#838383', '#6e6e6e', '#b5fbab', '#95f58b','#78f572', '#50f150','#1eb51e', '#0ea10e', '#1464d3', '#2883f1', '#50a5f5','#97d3fb', '#b5f1fb','#fffbab', '#ffe978', '#ffc13c', '#ffa100', '#ff6000','#ff3200', '#e11400','#c10000', '#a50000', '#870000', '#643c31', '#8d6558','#b58d83', '#c7a095','#f1ddd3', '#cecbdc'])#, '#aca0c7', '#9b89bd', '#725ca3','#695294', '#770077','#8d008d', '#b200b2', '#c400c4', '#db00db'])
    return cmap
colormap = weatherbell_precip_colormap()
colormap.set_over(color='#aca0c7')
bounds = [0,0.01,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.5,3,3.5,4,5,6,7,8,9,10]#,11,12,13,14,15,16,17,18,19,20])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))


# Load and filter master dataset
master_ds = xr.open_dataset(nc_file)
time_index = pd.DatetimeIndex(master_ds.time.values)
filtered_times = time_index[time_index.hour.isin([3, 6, 9, 12, 15, 18, 21, 0])]
master_ds = master_ds.sel(time=filtered_times).sortby('time')
master_ds = master_ds.sel(time=master_ds.time[start_time:end_time])
times = master_ds.time.values


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
fake_data = np.zeros((num_times, len(master_fine_lats), len(master_fine_lons)))
fake_coarse_data = np.zeros((num_times, len(master_coarse_lats), len(master_coarse_lons)))


# Load reference dataset
reference_ds = xr.load_dataset(reference_file, engine='cfgrib')
reference_ds = reference_ds.assign_coords(longitude=(((reference_ds.longitude + 180) % 360) - 180)).sortby('longitude')
reference_ds = reference_ds.sortby('latitude', ascending=True)


# Load grid domains
with open(grid_domains_file, 'rb') as f:
    grid_domains = pickle.load(f)


# Initialize model
model = UNetWithAttention(1, 1, output_shape=(64, 64)).to(device)


# Process each domain
start_lat_idx, end_lat_idx = 0, 64
start_lon_idx, end_lon_idx = 0, 64
start_coarse_lat_idx, end_coarse_lat_idx = 0, 16
start_coarse_lon_idx, end_coarse_lon_idx = 0, 16

for domain in tqdm(available_domains):
    min_lat, max_lat, min_lon, max_lon = grid_domains[domain]

    if not pad:
        cropped_reference_ds = reference_ds.sel(latitude=slice(min_lat, max_lat - 0.25), longitude=slice(min_lon, max_lon - 0.25))
        cropped_reference_ds_latitudes = cropped_reference_ds.latitude.values
        cropped_reference_ds_longitudes = cropped_reference_ds.longitude.values
        coarse_ds = master_ds.interp(lat=cropped_reference_ds_latitudes, lon=cropped_reference_ds_longitudes)
    else:
        cropped_input_reference_ds = reference_ds.sel(latitude=slice(min_lat-0.25, max_lat), longitude=slice(min_lon-0.25, max_lon))
        cropped_input_reference_ds_latitudes = cropped_input_reference_ds.latitude.values
        cropped_input_reference_ds_longitudes = cropped_input_reference_ds.longitude.values
        coarse_ds = master_ds.interp(lat=cropped_input_reference_ds_latitudes, lon=cropped_input_reference_ds_longitudes)


    checkpoint_path = f'checkpoints/best/{domain}_model.pt'
    if os.path.exists(checkpoint_path):
        print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        #print the test and bilinear loss
        percent_error_reduction = (1-(np.sqrt(checkpoint['test_loss'])/np.sqrt(checkpoint['bilinear_loss'])))*100
        print(f'{domain}: {percent_error_reduction:.2f}%')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        tp = torch.tensor(coarse_ds.tp.values, dtype=torch.float32).to(device)
        output = model(tp).cpu().detach().numpy()*0.0393701
        fake_data[:, start_lat_idx:end_lat_idx, start_lon_idx:end_lon_idx] = output
    else:
        fake_data[:, start_lat_idx:end_lat_idx, start_lon_idx:end_lon_idx] = np.nan
    fake_coarse_data[:, start_coarse_lat_idx:end_coarse_lat_idx, start_coarse_lon_idx:end_coarse_lon_idx] = coarse_ds.tp.values[:, 1:-1, 1:-1]*0.0393701

    start_lon_idx += 64
    end_lon_idx += 64
    start_coarse_lon_idx += 16
    end_coarse_lon_idx += 16
    if end_lon_idx > len(master_fine_lons):
        start_lon_idx, end_lon_idx = 0, 64
        start_lat_idx += 64
        end_lat_idx += 64
        start_coarse_lon_idx, end_coarse_lon_idx = 0, 16
        start_coarse_lat_idx += 16
        end_coarse_lat_idx += 16

fake_data_sum = np.sum(fake_data, axis=0)

# Determine the extent based on non-NaN data
non_nan_indices = np.where(~np.isnan(fake_data_sum))
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
    ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', alpha=0.75, linewidth=0.5)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    cf = ax.pcolormesh(master_fine_lons, master_fine_lats, fake_data[i], transform=ccrs.PlateCarree(), vmin=0, vmax=0.5)#, cmap=colormap, norm=norm)
    # cf = ax.contourf(master_fine_lons, master_fine_lats, fake_data[i], transform=ccrs.PlateCarree(), levels=bounds, cmap=colormap, norm=norm)
    plt.colorbar(cf, ax=ax, orientation='horizontal', label='tp (in)', pad=0.02)
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f'HARPNET Output {times[i]}', y=0.95)  # Increase the y parameter to move the title up
    plt.savefig(f'{fig_dir}/{i}.png', bbox_inches='tight')
    plt.close()

# Plot the sum with the calculated extent
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.coastlines()
ax.add_feature(cfeature.STATES.with_scale('10m'))
ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', alpha=0.75, linewidth=0.5)
ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
cf = ax.pcolormesh(master_fine_lons, master_fine_lats, fake_data_sum, transform=ccrs.PlateCarree())#, cmap=colormap, norm=norm)
# cf = ax.contourf(master_fine_lons, master_fine_lats, fake_data_sum, transform=ccrs.PlateCarree(), levels=bounds, cmap=colormap, norm=norm)
plt.colorbar(cf, ax=ax, orientation='horizontal', label='tp (in)', pad=0.02)
plt.subplots_adjust(top=0.9)
plt.suptitle(f'HARPNET Summed Output: {times[0]}-{times[-1]}')
plt.savefig(f'{fig_dir}/sum.png', bbox_inches='tight')
plt.close()

#create an xarray dataset from master_fine_lons, master_fine_lats, and fake_data
ds = xr.Dataset(
    {
        'tp': (['time', 'lat', 'lon'], fake_data)
    },
    coords={
        'time': times,
        'lat': master_fine_lats,
        'lon': master_fine_lons
    }
)

coarse_ds = xr.Dataset(
    {
        'tp': (['time', 'lat', 'lon'], fake_coarse_data)
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

echo_peak = (38.85, -120.08)
palisades_tahoe = (39.19, -120.27)
mt_rose = (39.32, -119.89)

# plot the cumulative precipitation at each point from ds as a time series
fig, ax = plt.subplots(figsize=(10, 5))
ds.tp.sel(lat=echo_peak[0], lon=echo_peak[1], method='nearest').plot(ax=ax, label='Echo Peak')
ds.tp.sel(lat=palisades_tahoe[0], lon=palisades_tahoe[1], method='nearest').plot(ax=ax, label='Palisades Tahoe')
ds.tp.sel(lat=mt_rose[0], lon=mt_rose[1], method='nearest').plot(ax=ax, label='Mt. Rose')
ax.set_xlabel('Time')
ax.set_ylabel('tp (in)')
ax.set_title('Cumulative Precipitation at Various Points')
ax.legend()
plt.show()

print(ds.tp.sel(lat=echo_peak[0], lon=echo_peak[1], method='nearest').values[-1], coarse_ds.tp.sel(lat=echo_peak[0], lon=echo_peak[1], method='nearest').values[-1])
print(ds.tp.sel(lat=palisades_tahoe[0], lon=palisades_tahoe[1], method='nearest').values[-1], coarse_ds.tp.sel(lat=palisades_tahoe[0], lon=palisades_tahoe[1], method='nearest').values[-1])
print(ds.tp.sel(lat=mt_rose[0], lon=mt_rose[1], method='nearest').values[-1], coarse_ds.tp.sel(lat=mt_rose[0], lon=mt_rose[1], method='nearest').values[-1])
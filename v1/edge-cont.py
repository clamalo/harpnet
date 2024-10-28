import os
import pickle
import numpy as np
import torch
import sys
import random
import warnings
warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Adjust the path if necessary
cwd_dir = (os.path.abspath(os.path.join(os.getcwd())))
sys.path.insert(0, cwd_dir)

# Import necessary modules
from utils.model import UNetWithAttention
from utils.utils import get_lats_lons, scale_coordinates
import utils.constants as constants

import xarray as xr
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

# Load grid domains
with open(f'{constants.domains_dir}grid_domains.pkl', 'rb') as f:
    grid_domains = pickle.load(f)

# Compute n_lat and n_lon
start_lat, start_lon = constants.start_lat, constants.start_lon
end_lat, end_lon = constants.end_lat, constants.end_lon

n_lat = (end_lat - start_lat) // 4
n_lon = (end_lon - start_lon) // 4

# Map domain indices to grid positions (lat_index, lon_index)
domain_positions = {}
trained_domains = []

for domain_index in grid_domains.keys():
    lat_index = domain_index // n_lon
    lon_index = domain_index % n_lon
    domain_positions[domain_index] = (lat_index, lon_index)

    # Check if checkpoint exists
    checkpoint_path = f'{constants.checkpoints_dir}best/{domain_index}_model.pt'
    if os.path.exists(checkpoint_path):
        trained_domains.append(domain_index)
    else:
        continue  # Skip domains without checkpoints

# Define test period
first_month = (1979, 10)
last_month = (1980, 9)
train_test_split = 0.2  # Use the same train/test split as during training

# Consistent random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load the entire dataset once
first_month_dt = pd.Timestamp(first_month[0], first_month[1], 1)
last_month_dt = pd.Timestamp(last_month[0], last_month[1], 1) + relativedelta(months=1) - pd.Timedelta(days=1)

# Assuming your data files are monthly NetCDF files named 'YYYY-MM.nc' in constants.nc_dir
all_files = [os.path.join(constants.nc_dir, f) for f in os.listdir(constants.nc_dir) if f.endswith('.nc')]
all_files.sort()

# Filter files within the date range
selected_files = []
for file in all_files:
    year_month = os.path.basename(file).split('.')[0]
    year, month = map(int, year_month.split('-'))
    file_date = pd.Timestamp(year, month, 1)
    if first_month_dt <= file_date <= last_month_dt:
        selected_files.append(file)

# Load the dataset
ds = xr.open_mfdataset(selected_files, combine='by_coords')

# Select test times based on the split
time_index = pd.DatetimeIndex(ds.time.values)
filtered_times = time_index[time_index.hour.isin([3, 6, 9, 12, 15, 18, 21, 0])]
ds = ds.sel(time=filtered_times)
ds = ds.sortby('time')

# Create train/test split indices
num_times = len(ds.time)
indices = np.arange(num_times)
np.random.shuffle(indices)
split_index = int(train_test_split * num_times)
test_indices = indices[:split_index]
# test_times = ds.time.isel(time=test_indices)
test_ds = ds.isel(time=test_indices)

# Now, process each domain
domain_outputs = {}
domain_positions_latlon = {}

# Reference grid for interpolation
reference_ds = xr.load_dataset(f'utils/reference_ds.grib2', engine='cfgrib')
reference_ds = reference_ds.assign_coords(longitude=(((reference_ds.longitude + 180) % 360) - 180)).sortby('longitude')
reference_ds = reference_ds.sortby('latitude', ascending=True)

for domain in tqdm(trained_domains, desc="Processing domains"):
    lat_index, lon_index = domain_positions[domain]
    min_lat, max_lat, min_lon, max_lon = grid_domains[domain]

    # Load the model
    checkpoint_path = f'{constants.checkpoints_dir}best/{domain}_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=constants.device)
    model = UNetWithAttention(1, 1, output_shape=(64,64)).to(constants.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepare the data for this domain
    # Interpolate and extract the input and target data
    # Input data is the coarse data interpolated to 0.25-degree grid
    # Target data is the high-resolution data interpolated to fine grid

    # Coarse grid coordinates
    coarse_lats = np.arange(min_lat, max_lat, 0.25)
    coarse_lons = np.arange(min_lon, max_lon, 0.25)

    # Fine grid coordinates
    fine_lats = scale_coordinates(coarse_lats, constants.scale_factor)
    fine_lons = scale_coordinates(coarse_lons, constants.scale_factor)

    # Extract data for the domain
    domain_test_ds = test_ds.sel(lat=slice(min_lat-0.25, max_lat+0.25), lon=slice(min_lon-0.25, max_lon+0.25))

    # Interpolate to coarse grid
    coarse_ds = domain_test_ds.interp(lat=coarse_lats, lon=coarse_lons, method='nearest')
    # Interpolate to fine grid
    fine_ds = domain_test_ds.interp(lat=fine_lats, lon=fine_lons, method='nearest')

    # Get the data arrays
    inputs = coarse_ds.tp.values.astype('float32')  # Shape: (num_samples, len(coarse_lats), len(coarse_lons))
    targets = fine_ds.tp.values.astype('float32')   # Shape: (num_samples, len(fine_lats), len(fine_lons))

    # Move data to torch tensors
    inputs_tensor = torch.from_numpy(inputs)
    targets_tensor = torch.from_numpy(targets)

    # Since the model expects inputs of shape (batch_size, H, W), ensure the shapes are correct
    inputs_tensor = inputs_tensor.to(constants.device)
    targets_tensor = targets_tensor.to(constants.device)

    # Get the model outputs
    outputs_list = []
    with torch.no_grad():
        for i in range(0, len(inputs_tensor), constants.operational_batch_size):
            batch_inputs = inputs_tensor[i:i+constants.operational_batch_size]
            batch_inputs = batch_inputs.to(constants.device)
            outputs = model(batch_inputs)
            outputs_list.append(outputs.cpu().numpy())

    outputs_array = np.concatenate(outputs_list, axis=0)  # Shape: (num_samples, 64, 64)

    # Store the outputs
    domain_outputs[domain] = outputs_array
    # Also store positions for visualization
    domain_positions_latlon[domain] = (min_lat, max_lat, min_lon, max_lon)

# Now, compute continuity errors between adjacent tiles
border_errors = {}

for domain in trained_domains:
    lat_index, lon_index = domain_positions[domain]
    outputs_A = domain_outputs[domain]

    # Check right neighbor (domain B)
    if lon_index < n_lon - 1:
        neighbor_domain_index = domain + 1  # Right neighbor in the domain list
        if neighbor_domain_index in trained_domains:
            outputs_B = domain_outputs[neighbor_domain_index]
            # Compute the difference along the shared border (rightmost column of A and leftmost column of B)
            # outputs_A: (num_samples, 64, 64)
            # outputs_B: (num_samples, 64, 64)
            # Get the rightmost column of A
            right_column_A = outputs_A[:, :, -1]  # Shape: (num_samples, 64)
            # Get the leftmost column of B
            left_column_B = outputs_B[:, :, 0]  # Shape: (num_samples, 64)

            # Compute the MSE along the border for each sample
            border_diff = right_column_A - left_column_B  # Shape: (num_samples, 64)
            mse = np.mean(border_diff ** 2)
            # Store the error
            border_errors[(domain, neighbor_domain_index)] = mse
        else:
            border_errors[(domain, 'right')] = np.nan

    # Check bottom neighbor (domain B)
    if lat_index < n_lat - 1:
        neighbor_domain_index = domain + n_lon  # Bottom neighbor in the domain list
        if neighbor_domain_index in trained_domains:
            outputs_B = domain_outputs[neighbor_domain_index]
            # Get the bottom row of A
            bottom_row_A = outputs_A[:, -1, :]  # Shape: (num_samples, 64)
            # Get the top row of B
            top_row_B = outputs_B[:, 0, :]  # Shape: (num_samples, 64)

            # Compute the MSE along the border for each sample
            border_diff = bottom_row_A - top_row_B  # Shape: (num_samples, 64)
            mse = np.mean(border_diff ** 2)
            # Store the error
            border_errors[(domain, neighbor_domain_index)] = mse
        else:
            border_errors[(domain, 'bottom')] = np.nan

# Visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([start_lon, end_lon, start_lat, end_lat], ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)

# Prepare a colormap
valid_errors = [mse for mse in border_errors.values() if not np.isnan(mse)]
if valid_errors:
    norm = mcolors.Normalize(vmin=min(valid_errors), vmax=max(valid_errors))
else:
    norm = mcolors.Normalize(vmin=0, vmax=1)  # Default range if no valid errors
cmap = plt.cm.viridis

for (domain_A, domain_B), mse in border_errors.items():
    if np.isnan(mse):
        continue

    min_lat_A, max_lat_A, min_lon_A, max_lon_A = domain_positions_latlon[domain_A]

    if isinstance(domain_B, int):
        min_lat_B, max_lat_B, min_lon_B, max_lon_B = domain_positions_latlon[domain_B]
    else:
        continue

    # For left-right borders
    if min_lon_B == max_lon_A and min_lat_B == min_lat_A:
        # Vertical line along the shared border at longitude max_lon_A (which should equal min_lon_B)
        lats = np.linspace(min_lat_A, max_lat_A, 100)
        lon = max_lon_A
        ax.plot([lon]*len(lats), lats, color=cmap(norm(mse)), linewidth=4)
    # For top-bottom borders
    elif min_lat_B == max_lat_A and min_lon_B == min_lon_A:
        # Horizontal line along the shared border at latitude max_lat_A (which should equal min_lat_B)
        lons = np.linspace(min_lon_A, max_lon_A, 100)
        lat = max_lat_A
        ax.plot(lons, [lat]*len(lons), color=cmap(norm(mse)), linewidth=4)

# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Continuity Error (MSE)')

plt.title('Tile-to-Tile Continuity Error (MSE) Between Adjacent Tiles')
plt.show()

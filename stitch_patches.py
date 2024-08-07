#can also plot performance by changing the shading of the fake data to be the test loss

from utils.utils3 import *
import matplotlib.pyplot as plt
import cartopy.crs as ccrs 

#create stitch figure direcotry if it doesnt exist
if not os.path.exists('figures/stitch'):
    os.makedirs('figures/stitch')

start_time = 104
end_time = 136
num_times = end_time - start_time
year,month = 2020, 3
master_ds = xr.open_dataset(f'{constants.nc_dir}{year}-{month:02d}.nc')
time_index = pd.DatetimeIndex(master_ds.time.values)
filtered_times = time_index[time_index.hour.isin([3, 6, 9, 12, 15, 18, 21, 0])]
master_ds = master_ds.sel(time=filtered_times)
master_ds = master_ds.sortby('time')
master_ds['days'] = master_ds.time.dt.dayofyear
#select the first 8 times
master_ds = master_ds.sel(time=master_ds.time[start_time:end_time])


available_domains = range(0,49)

master_fine_lats = set()
master_fine_lons = set()

for domain in available_domains:
    fine_lats, fine_lons, cropped_reference_ds_latitudes, cropped_reference_ds_longitudes = get_lats_lons(domain, pad=False)
    master_fine_lats.update(fine_lats)
    master_fine_lons.update(fine_lons)

# Convert sets to numpy arrays
master_fine_lats = np.array(list(master_fine_lats))
master_fine_lons = np.array(list(master_fine_lons))
#sort the lats and lons
master_fine_lats.sort()
master_fine_lons.sort()
fake_data = np.zeros((num_times, len(master_fine_lats), len(master_fine_lons)))


reference_ds = xr.load_dataset(f'{constants.base_dir}reference_ds.grib2', engine='cfgrib')
reference_ds = reference_ds.assign_coords(longitude=(((reference_ds.longitude + 180) % 360) - 180)).sortby('longitude')
reference_ds = reference_ds.sortby('latitude', ascending=True)

with open(f'{constants.domains_dir}grid_domains.pkl', 'rb') as f:
    grid_domains = pickle.load(f)

from utils.model import UNetWithAttention
model = UNetWithAttention(1, 1, output_shape=(64,64)).to('mps')


start_lon_idx, end_lon_idx = 0, 64
start_lat_idx, end_lat_idx = 0, 64
#lon idxs need to reset after hitting the end of master_fine_lons
for domain in tqdm(available_domains):

    min_lat, max_lat, min_lon, max_lon = grid_domains[domain]

    cropped_reference_ds = reference_ds.sel(latitude=slice(min_lat, max_lat-0.25), longitude=slice(min_lon, max_lon-0.25))
    cropped_reference_ds_latitudes = cropped_reference_ds.latitude.values
    cropped_reference_ds_longitudes = cropped_reference_ds.longitude.values
    coarse_ds = master_ds.interp(lat=cropped_reference_ds_latitudes, lon=cropped_reference_ds_longitudes)
    
    if os.path.exists(f'checkpoints/{domain}/0_model.pt'):
        # fake_data[start_lat_idx:end_lat_idx, start_lon_idx:end_lon_idx] = domain
        checkpoint = torch.load(f'checkpoints/{domain}/0_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        tp = torch.tensor(coarse_ds.tp.values)
        #turn to tensor
        tp = torch.tensor(tp, dtype=torch.float32).to('mps')
        output = model(tp)
        fake_data[:, start_lat_idx:end_lat_idx, start_lon_idx:end_lon_idx] = output.cpu().detach().numpy()
    else:
        fake_data[:, start_lat_idx:end_lat_idx, start_lon_idx:end_lon_idx] = np.nan

    start_lon_idx += 64
    end_lon_idx += 64
    if end_lon_idx > len(master_fine_lons):
        start_lon_idx, end_lon_idx = 0, 64
        start_lat_idx += 64
        end_lat_idx += 64


for i in range(num_times):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cartopy.feature.STATES.with_scale('10m'))
    cf = ax.pcolormesh(master_fine_lons, master_fine_lats, fake_data[i], transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=10)
    plt.colorbar(cf, ax=ax, orientation='horizontal', label='Domain')
    #shrink white space
    plt.tight_layout()
    plt.savefig(f'figures/stitch/{i}.png', bbox_inches='tight')
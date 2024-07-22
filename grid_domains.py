import xarray as xr
import pickle

def create_grid_domains():
    ds = xr.open_dataset('/Volumes/T9/monthly/1979-10.nc', chunks={'time': 100})
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
            cropped_ds = ds.isel(lat=slice(lat_idx, lat_idx+64), lon=slice(lon_idx, lon_idx+64))
            grid_domains[total_domains] = [lat_idx, lat_idx+64, lon_idx, lon_idx+64]
            total_domains += 1
    with open('grid_domains.pkl', 'wb') as f:
        pickle.dump(grid_domains, f)
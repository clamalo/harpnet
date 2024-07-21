import numpy as np
import xarray as xr
import os
from dask.diagnostics import ProgressBar
from torch.utils.data import Dataset, DataLoader

max_lat, min_lat, max_lon, min_lon = 45, 35, -105, -115  # Utah

def unnormalize_data(normalized_data, key, mean_std_dict):
    mean = mean_std_dict[key]['mean']
    std = mean_std_dict[key]['std']
    return (normalized_data * std) + mean

def output_to_dataset(output, ds_latitudes, ds_longitudes, flip):
    if flip:
        output = np.flip(output, axis=1)
    times = np.arange(output.shape[0])
    tp_da = xr.DataArray(output, dims=("time", "latitude", "longitude"), coords={"time": times, "latitude": ds_latitudes, "longitude": ds_longitudes})
    tp_ds = xr.Dataset({"tp": tp_da})
    return tp_ds


    
def load_fine_dataset(path, months):
    file_paths = [os.path.join(path, fp) for fp in os.listdir(path) if fp.endswith('.nc')]
    file_paths = sorted(file_paths, key=lambda fp: os.path.basename(fp))
    with ProgressBar():
        ds = xr.open_mfdataset(file_paths, combine='by_coords', parallel=True, chunks={'Time': 100000})
        ds = ds.assign_coords(hour=ds.Time.dt.hour, day=ds.Time.dt.dayofyear)
    ds = ds.sortby('Time')
    ds = ds.rename({'Time': 'time'})
    ds['days'] = ds.time.dt.dayofyear
    for var in ds.data_vars:
        ds[var] = ds[var].astype('float32')
    return ds



def load_array(arr, shapes, indices, BASE_DIR, target=False):
    print(f'Loading {arr.name}...')
    with ProgressBar():
        var_arr = arr.values
        if len(var_arr.shape) == 3:
            var_arr = np.expand_dims(var_arr, axis=1)
    var_arr = var_arr[indices]
    if target:
        save_name = 'target'
    else:
        save_name = arr.name
    shapes[f'{save_name}_train'] = var_arr.shape
    np.memmap(os.path.join(BASE_DIR, f'{save_name}_mmap_train.dat'), dtype='float32', mode='w+', shape=var_arr.shape)[...] = var_arr



def load_array_test(arr, shapes, indices, BASE_DIR, target=False):
    print(f'Loading {arr.name}...')
    with ProgressBar():
        var_arr = arr.values
        if len(var_arr.shape) == 3:
            var_arr = np.expand_dims(var_arr, axis=1)
    var_arr = var_arr[indices]
    if target:
        save_name = 'target'
    else:
        save_name = arr.name
    shapes[f'{save_name}_test'] = var_arr.shape
    np.memmap(os.path.join(BASE_DIR, f'{save_name}_mmap_test.dat'), dtype='float32', mode='w+', shape=var_arr.shape)[...] = var_arr



class MemMapDataset(Dataset):
    def __init__(self, data, labels, days):
        self.data = data
        self.labels = labels
        self.days = days
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.days[idx]
    


def oper_prep_gfs(frame):
    surface_gfs_ds = xr.open_dataset(f'data/oper/gfs.f{frame:03d}.grib', engine='cfgrib', filter_by_keys={'typeOfLevel': 'surface'})
    single_layer_gfs_ds = xr.open_dataset(f'data/oper/gfs.f{frame:03d}.grib', engine='cfgrib', filter_by_keys={'typeOfLevel': 'atmosphereSingleLayer'})
    gfs_ds = xr.open_dataset(f'data/oper/gfs.f{frame:03d}.grib', engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'}).rename({'isobaricInhPa': 'level', 'gh': 'z'})
    gfs_ds['z'] = gfs_ds['z'] * 9.80665
    gfs_ds['tp'] = surface_gfs_ds['tp']
    gfs_ds['cape'] = surface_gfs_ds['cape']
    gfs_ds['tcw'] = single_layer_gfs_ds['pwat']
    gfs_ds = gfs_ds.assign_coords(longitude=(((gfs_ds.longitude + 180) % 360) - 180)).sortby('longitude')
    gfs_ds = gfs_ds.sortby('latitude', ascending=False)
    gfs_ds = gfs_ds.sel(latitude=slice(max_lat-1, min_lat+1), longitude=slice(min_lon+1, max_lon-1))

    for var in gfs_ds.data_vars:
        gfs_ds[var] = gfs_ds[var].astype('float32')

    return gfs_ds
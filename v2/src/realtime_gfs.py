import requests
import xarray as xr
from tqdm import tqdm
import os

def realtime_gfs(datestr, cycle, frames, ingest):

    def ingest_gfs_data(datestr, cycle, frame, file_path):
        # idx_url = f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{datestr}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{frame:03d}.idx'
        # idx_response = requests.get(idx_url)
        # idx_lines = idx_response.text.split('\n')
        # start_byte = idx_lines[595].split(':')[1]
        # end_byte = idx_lines[596].split(':')[1]
        # gfs_url = f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{datestr}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{frame:03d}'
        # r = requests.get(gfs_url, headers={'Range': f'bytes={start_byte}-{end_byte}'}, allow_redirects=True)

        gfs_url = f'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?dir=%2Fgfs.{datestr}%2F{cycle}%2Fatmos&file=gfs.t{cycle}z.pgrb2.0p25.f{frame:03d}&var_APCP=on&lev_surface=on'
        r = requests.get(gfs_url, allow_redirects=True)
        
        # Append the content to the same file
        with open(file_path, 'ab') as f:
            f.write(r.content)
    
    if ingest:
        single_file_path = 'data/gfs.grib'
        
        if os.path.exists(single_file_path):
            os.remove(single_file_path)
        
        for frame in tqdm(frames):
            ingest_gfs_data(datestr, cycle, frame, single_file_path)
    

    ds = xr.open_dataset('data/gfs.grib', engine='cfgrib')
    ds = ds.sortby('valid_time')
    for t in range(len(ds.valid_time)):
        if ds['valid_time'][t].dt.hour in [0, 6, 12, 18]:
            ds['tp'][t] = ds['tp'][t] - ds['tp'][t-1]
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds.sortby('lat', ascending=True)
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
    times = ds.valid_time.values

    return ds

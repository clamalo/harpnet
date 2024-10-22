from ecmwf.opendata import Client
import xarray as xr
import os

def realtime_ecmwf(datestr, cycle, frames, ingest):

    single_file_path = 'data/ecmwf.grib'
    
    if ingest:

        if os.path.exists(single_file_path):
            os.remove(single_file_path)

        client = Client("ecmwf", beta=True)
        parameters = ['tp']
        client.retrieve(
            date=datestr,
            time=cycle,
            step=frames,
            stream="oper",
            type="fc",
            levtype="sfc",
            param=parameters,
            target=single_file_path
        )

    ds = xr.open_dataset(f'data/ecmwf.grib', engine='cfgrib')
    ds = ds.sortby('valid_time')
    ds['cum_tp'] = ds['tp']*1000
    ds['tp'][0] = ds['cum_tp'][0]
    for t in range(1,len(ds.valid_time)):
        ds['tp'][t] = ds['cum_tp'][t]-ds['cum_tp'][t-1]
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds.sortby('lat', ascending=True)
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')

    return ds
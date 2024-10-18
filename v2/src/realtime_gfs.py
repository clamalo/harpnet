import asyncio
import aiohttp
import aiofiles
import xarray as xr
from tqdm.asyncio import tqdm as atqdm
import os

def realtime_gfs(datestr, cycle, frames, ingest):
    async def fetch(session, url):
        async with session.get(url) as response:
            response.raise_for_status()  # Ensure we notice bad responses
            return await response.text()

    async def download(session, url, headers, sem):
        async with sem:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.read()

    async def ingest_gfs_data(session, datestr, cycle, frame, file_path, sem):
        idx_url = f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{datestr}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{frame:03d}.idx'
        try:
            idx_text = await fetch(session, idx_url)
            idx_lines = idx_text.strip().split('\n')
            
            # Ensure that there are enough lines in the index file
            if len(idx_lines) < 597:
                raise ValueError(f"Index file for frame {frame} is incomplete.")
            
            start_byte = idx_lines[595].split(':')[1]
            end_byte = idx_lines[596].split(':')[1]
            gfs_url = f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{datestr}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{frame:03d}'
            
            headers_range = {'Range': f'bytes={start_byte}-{end_byte}'}
            content = await download(session, gfs_url, headers_range, sem)
            
            # Append the content to the same file asynchronously
            async with aiofiles.open(file_path, 'ab') as f:
                await f.write(content)
                
        except Exception as e:
            print(f"Error ingesting frame {frame}: {e}")

    async def main_ingest(datestr, cycle, frames):
        single_file_path = os.path.join('data', 'gfs.grib')
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(single_file_path), exist_ok=True)
        
        # Remove existing file if it exists
        if os.path.exists(single_file_path):
            os.remove(single_file_path)
        
        sem = asyncio.Semaphore(10)  # Adjust concurrency as needed
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                ingest_gfs_data(session, datestr, cycle, frame, single_file_path, sem)
                for frame in frames
            ]
            # Use tqdm for progress bar
            for f in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading frames"):
                await f

    # Run the asynchronous ingest
    if ingest:
        asyncio.run(main_ingest(datestr, cycle, frames))

    # Load the dataset using xarray
    ds = xr.open_dataset('data/gfs.grib', engine='cfgrib')
    ds = ds.sortby('valid_time')
    
    # Adjust 'tp' variable if it exists
    if 'tp' in ds.variables:
        tp = ds['tp'].values
        for t in range(1, len(ds.valid_time)):
            if ds['valid_time'][t].dt.hour in [0, 6, 12, 18]:
                ds['tp'][t] = tp[t] - tp[t-1]
    
    # Rename and sort coordinates
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds.sortby('lat', ascending=True)
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
    times = ds.valid_time.values

    return ds

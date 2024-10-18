import asyncio
import aiohttp
import pandas as pd
import aiofiles
from tqdm.asyncio import tqdm as atqdm
from io import StringIO
import os
import xarray as xr


def realtime_gefs(datestr, cycle, frames, ingest):
    async def fetch(session, url):
        async with session.get(url) as response:
            response.raise_for_status()  # Ensure we notice bad responses
            return await response.text()

    async def download(session, url, start, end, sem):
        async with sem:
            headers = {"Range": f"bytes={start}-{end}"} if end != -1 else {}
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.read()

    async def parse_gefs_idx(session, idx_link):
        r = await fetch(session, idx_link)
        data = StringIO(r)
        idx_df = pd.read_csv(data, sep=":", header=None)
        idx_df.columns = ["line", "start", "datestr", "param", "level", "forecast_step", "member"]
        idx_df["end"] = idx_df["start"].shift(-1).fillna(-1).astype(int)
        idx_df["link"] = idx_link.replace('.idx','')
        idx_df = idx_df[["line", "start", "end", "datestr", "param", "level", "forecast_step", "member", "link"]]
        idx_df.drop(columns=["datestr", "line", "forecast_step", "member"], inplace=True)
        return idx_df

    async def download_gefs(date, cycle, steps, req_param, req_level, output_file):
        ROOT = "https://noaa-gefs-pds.s3.amazonaws.com"
        members = [f"p{str(i).zfill(2)}" for i in range(1, 31)] + ["c00"]
        df_list = []

        sem = asyncio.Semaphore(25)  # Adjust semaphore as needed for concurrency

        async with aiohttp.ClientSession() as session:
            # Prepare all idx links for all members and all steps
            idx_tasks = []
            for step in steps:
                for member in members:
                    if req_param == ['APCP']:
                        host = f"{ROOT}/gefs.{date}/{cycle}/atmos/pgrb2sp25"
                        idx_link = f"{host}/ge{member}.t{cycle}z.pgrb2s.0p25.f{step:03}.idx"
                    else:
                        host = f"{ROOT}/gefs.{date}/{cycle}/atmos/pgrb2ap5"
                        idx_link = f"{host}/ge{member}.t{cycle}z.pgrb2a.0p50.f{step:03}.idx"
                    idx_tasks.append(parse_gefs_idx(session, idx_link))
            
            # Parse all idx files
            parsed_idx = await asyncio.gather(*idx_tasks)
            df_list.extend(parsed_idx)

            idx_df = pd.concat(df_list, ignore_index=True)
            if req_level is None:
                idx_df = idx_df[idx_df.param.isin(req_param)]
            else:
                idx_df = idx_df[
                    (idx_df.param.isin(req_param)) &
                    (idx_df.level.isin(req_level))
                ]
            idx = idx_df.to_dict(orient="records")

            # Prepare download tasks
            download_tasks = [
                download(session, record['link'], record["start"], record["end"], sem)
                for record in idx
            ]

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Open the output file once and write all data
            # Use 'wb' for the first write and 'ab' for appending
            async with aiofiles.open(output_file, 'wb') as f:
                with atqdm(total=len(download_tasks), desc="Downloading data") as pbar:
                    for task in asyncio.as_completed(download_tasks):
                        try:
                            content = await task
                            await f.write(content)
                        except Exception as e:
                            print(f"Error downloading a chunk: {e}")
                        pbar.update(1)

    async def main_download(date, cycle, steps):
        req_param = ["APCP"]  # Modify as needed
        req_level = None      # Modify as needed or set to a list like ["1000 mb", "925 mb"]
        output_file = os.path.join(os.getcwd(), 'data', f'gefs.grib')

        await download_gefs(date, cycle, steps, req_param, req_level, output_file)

        # If you have additional parameters and levels, uncomment and modify accordingly
        # req_param = ["HGT", "TMP"]
        # req_level = ["1000 mb", "925 mb", "850 mb", "700 mb", "500 mb"]
        # output_file = os.path.join(os.getcwd(), 'data', f'p_gefs.grib2')
        # await download_gefs(date, cycle, steps, req_param, req_level, output_file)

    # Run the asynchronous download
    if ingest:
        if os.path.exists('data/gefs.grib'):
            os.remove('data/gefs.grib')
        asyncio.run(main_download(datestr, cycle, frames))


    ds = xr.open_dataset('data/gefs.grib', engine='cfgrib', filter_by_keys={'dataType': 'pf'})
    ds = ds.sortby('number')
    ds = ds.sortby('valid_time')
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds.sortby('lat', ascending=True)
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
    times = ds.valid_time.values

    return ds
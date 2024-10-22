import asyncio
import aiohttp
import pandas as pd
import aiofiles
from tqdm.asyncio import tqdm as atqdm
import os
import xarray as xr
import json


def realtime_eps(datestr, cycle, frames, ingest):
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

    async def parse_idx(session, idx_link):
        r = await fetch(session, idx_link)
        idx = [json.loads(line) for line in r.splitlines()]
        idx_df = pd.DataFrame(idx)
        idx_df.drop(columns=["domain", "stream", "date", "time", "expver", "class", "step"], inplace=True)
        idx_df["member"] = idx_df["number"].apply(lambda x: f"p{int(x):02d}" if not pd.isna(x) else "c00")
        idx_df.drop(columns=["number"], inplace=True)
        idx_df["level"] = idx_df.apply(
            lambda x: x["levtype"] if x["levtype"] == "sfc" else x["levelist"] + " hPa", axis=1
        )
        idx_df.drop(columns=["levtype", "levelist"], inplace=True)
        idx_df.rename(columns={"_offset": "start"}, inplace=True)
        idx_df["end"] = idx_df["start"] + idx_df["_length"]
        return idx_df

    async def download_eps(date, cycle, steps, req_param, req_level, output_file):
        ROOT = 'https://data.ecmwf.int/forecasts'
        host = f"{ROOT}/{date}/{cycle}z/ifs/0p25/enfo"

        sem = asyncio.Semaphore(20)  # Adjust as needed

        idx_df_list = []

        async with aiohttp.ClientSession() as session:
            idx_tasks = []
            step_list = []
            for step in steps:
                idx_link = f"{host}/{date}{cycle}0000-{step}h-enfo-ef.index"
                idx_tasks.append(parse_idx(session, idx_link))
                step_list.append(step)

            # Parse all index files concurrently
            idx_dfs = await asyncio.gather(*idx_tasks)

            # For each idx_df, add the corresponding pgrb2_link and step
            for idx_df, step in zip(idx_dfs, step_list):
                pgrb2_link = f"{host}/{date}{cycle}0000-{step}h-enfo-ef.grib2"
                idx_df['pgrb2_link'] = pgrb2_link
                idx_df['step'] = step  # Add step for reference if needed
                idx_df_list.append(idx_df)

            # Concatenate all index DataFrames
            idx_df = pd.concat(idx_df_list, ignore_index=True)

            # Filter according to requested parameters and levels
            if req_level is None:
                idx_df = idx_df[idx_df.param.isin(req_param)]
            else:
                idx_df = idx_df[
                    idx_df.param.isin(req_param) & idx_df.level.isin(req_level)
                ]

            idx = idx_df.to_dict(orient="records")

            # Prepare download tasks for all chunks
            tasks = [
                download(session, record['pgrb2_link'], record["start"], record["end"], sem)
                for record in idx
            ]

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Open the output file and write all data
            async with aiofiles.open(output_file, 'wb') as f:
                with atqdm(total=len(tasks), desc="Downloading data") as pbar:
                    for task in asyncio.as_completed(tasks):
                        try:
                            content = await task
                            await f.write(content)
                        except Exception as e:
                            print(f"Error downloading a chunk: {e}")
                        pbar.update(1)

    async def main_download(date, cycle, steps):
        req_param = ["tp"]  # Modify as needed
        req_level = None    # Modify as needed or set to a list like ["1000 hPa", "925 hPa"]
        output_file = os.path.join(os.getcwd(), 'data', f'eps.grib')

        await download_eps(date, cycle, steps, req_param, req_level, output_file)

    # Run the asynchronous download
    if ingest:
        if os.path.exists('data/eps.grib'):
            os.remove('data/eps.grib')
        asyncio.run(main_download(datestr, cycle, frames))

    # Load the dataset using xarray and cfgrib
    ds = xr.open_dataset('data/eps.grib', engine='cfgrib', filter_by_keys={'dataType': 'pf'})
    ds = ds.sortby('number')
    ds = ds.sortby('valid_time')
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds.sortby('lat', ascending=True)
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
    times = ds.valid_time.values

    return ds
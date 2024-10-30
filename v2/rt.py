import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from tqdm import tqdm
import numpy as np
from metpy.plots import USCOUNTIES
import xarray as xr
import os

from src.model import UNetWithAttention
from src.constants import TORCH_DEVICE
model = UNetWithAttention(1, 1, output_shape=(64,64)).to(TORCH_DEVICE)
from src.get_coordinates import get_coordinates
from src.realtime_ecmwf import realtime_ecmwf
from src.realtime_gfs import realtime_gfs
from src.realtime_gefs import realtime_gefs
from src.realtime_eps import realtime_eps



datestr, cycle = '20241029', '00'
frames = list(range(3, 145, 3))
ingest = False
rt_model = 'gfs'



if rt_model == 'ecmwf':
    ds = realtime_ecmwf(datestr, cycle, frames, ingest)
    members = range(1)
if rt_model == 'gfs':
    ds = realtime_gfs(datestr, cycle, frames, ingest)
    members = range(1)
if rt_model == 'gefs':
    ds = realtime_gefs(datestr, cycle, frames, ingest)
    members = range(30)
if rt_model == 'eps':
    ds = realtime_eps(datestr, cycle, frames, ingest)
    members = range(50)



tile_data = {}
valid_tiles = list(range(36))
for tile in valid_tiles:

    coarse_lats_pad, coarse_lons_pad, coarse_lats, coarse_lons, fine_lats, fine_lons = get_coordinates(tile)

    # if not os.path.exists(f'best/{tile}_model.pt') or tile not in [12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28]:
    if not os.path.exists(f'best/{tile}_model.pt') or tile not in [15, 16, 21, 22]:
        output = np.full((len(members), len(frames), len(fine_lats), len(fine_lons)), np.nan)

    else:
        if rt_model == 'ecmwf':
            coarse_ds = ds.interp(lat=coarse_lats_pad, lon=coarse_lons_pad)
        if rt_model == 'gfs':
            coarse_ds = ds.interp(lat=coarse_lats_pad, lon=coarse_lons_pad)
        if rt_model == 'gefs':
            coarse_ds = ds.interp(lat=coarse_lats_pad, lon=coarse_lons_pad)
            coarse_ds = coarse_ds.transpose('step', 'number', 'lat', 'lon')
            for t in range(len(coarse_ds.valid_time)):
                if coarse_ds['valid_time'][t].dt.hour in [0, 6, 12, 18]:
                    coarse_ds['tp'][t] = coarse_ds['tp'][t] - coarse_ds['tp'][t-1]
            coarse_ds = coarse_ds.transpose('number', 'step', 'lat', 'lon')
        if rt_model == 'eps':
            coarse_member_datasets = []
            for member in tqdm(range(len(ds.number))):
                member_ds = ds.isel(number=member)
                coarse_member_dataset = member_ds.interp(lat=coarse_lats_pad, lon=coarse_lons_pad)
                coarse_member_datasets.append(coarse_member_dataset)
            coarse_ds = xr.concat(coarse_member_datasets, dim='number')
            coarse_ds = coarse_ds.transpose('step', 'number', 'lat', 'lon')
            coarse_ds['cum_tp'] = coarse_ds['tp']*1000
            coarse_ds['tp'][0] = coarse_ds['cum_tp'][0]
            for t in range(1,len(coarse_ds.valid_time)):
                coarse_ds['tp'][t] = coarse_ds['cum_tp'][t]-coarse_ds['cum_tp'][t-1]
            coarse_ds = coarse_ds.transpose('number', 'step', 'lat', 'lon')


        tp = coarse_ds.tp.values
        

        if len(tp.shape) == 3:
            tp = np.expand_dims(tp, axis=0)

        tensor_tp = torch.tensor(tp, dtype=torch.float32).to(TORCH_DEVICE)

        checkpoint = torch.load(f'best/{tile}_model.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        with torch.no_grad():

            output = []
            for m in tqdm(range(0, tp.shape[0], 1)):
                input_arr = tensor_tp[m]

                m_output = model(input_arr).cpu().detach().numpy()
                output.append(np.expand_dims(m_output, axis=0))

        output = np.concatenate(output, axis=0)

    tile_ds = xr.Dataset(
        {
            'tp': (['number', 'step', 'lat', 'lon'], output),
        },
        coords={
            'lat': fine_lats,
            'lon': fine_lons,
            'step': frames,
            'number': members
        }
    )
    tile_data[tile] = tile_ds

combined_ds = xr.combine_by_coords(list(tile_data.values()))

# crop combined_ds to where the tp values are not nan
valid_mask = ~combined_ds.tp.isnull().all(dim=['number', 'step'])
valid_lats = combined_ds.lat.where(valid_mask.any(dim='lon'), drop=True)
valid_lons = combined_ds.lon.where(valid_mask.any(dim='lat'), drop=True)
combined_ds = combined_ds.sel(lat=valid_lats, lon=valid_lons)


import matplotlib.colors as colors
def weatherbell_precip_colormap():
    cmap = colors.ListedColormap(['#ffffff', '#bdbfbd','#aba5a5', '#838383', '#6e6e6e', '#b5fbab', '#95f58b','#78f572', '#50f150','#1eb51e', '#0ea10e', '#1464d3', '#2883f1', '#50a5f5','#97d3fb', '#b5f1fb','#fffbab', '#ffe978', '#ffc13c', '#ffa100', '#ff6000','#ff3200', '#e11400','#c10000', '#a50000', '#870000', '#643c31', '#8d6558','#b58d83', '#c7a095','#f1ddd3', '#cecbdc'])#, '#aca0c7', '#9b89bd', '#725ca3','#695294', '#770077','#8d008d', '#b200b2', '#c400c4', '#db00db'])
    cmap.set_over(color='#aca0c7')
    bounds = [0,0.01,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.5,3,3.5,4,5,6,7,8,9,10]#,11,12,13,14,15,16,17,18,19,20])
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
    return cmap, norm, bounds



# for t in tqdm(range(len(combined_ds.step))):
#     time_ds = combined_ds.isel(step=t)
#     fig, axes = plt.subplots(5, 6, figsize=(30, 25), subplot_kw={'projection': ccrs.PlateCarree()})
#     axes = axes.flatten()
#     # for member in range(output.shape[0]):
#     for member in range(30):
#         ax = axes[member]
#         cf = ax.pcolormesh(
#             time_ds.lon, time_ds.lat,
#             time_ds.tp[member].values*0.0393701,
#             transform=ccrs.PlateCarree(), 
#             cmap=weatherbell_precip_colormap()[0], 
#             norm=weatherbell_precip_colormap()[1])
#         ax.add_feature(cartopy.feature.STATES, linewidth=0.5)
#         ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', linewidth=0.5)
#         ax.set_title(f'Member {member + 1}', fontsize=12)
#         # ax extent to central western colorado
#         ax.set_extent([-109.5, -105, 37.5, 41.5])
#     for idx in range(output.shape[0], len(axes)):
#         fig.delaxes(axes[idx])
#     cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])  # [left, bottom, width, height]
#     fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', label='Precipitation (inches)')
#     valid_time = coarse_ds.valid_time[t].values
#     if isinstance(valid_time, np.datetime64):
#         valid_time_str = np.datetime_as_string(valid_time, unit='h')
#     else:
#         valid_time_str = str(valid_time)
#     fig.suptitle(f'Precipitation Forecast at {valid_time_str}', fontsize=16)
#     plt.tight_layout(rect=[0, 0.1, 1, 0.95])
#     plt.savefig(f'figures/rt/precipitation_t{int(t*3):03d}.png', dpi=300)
#     plt.close(fig)


# cum sum
total_ds = combined_ds.cumsum(dim='step')

for t in range(len(total_ds.step)):
    time_ds = combined_ds.isel(step=t).mean(dim='number')
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    cf = ax.pcolormesh(time_ds['lon'], time_ds['lat'], time_ds['tp']*0.0393701, vmin=0, vmax=0.5)#, transform=ccrs.PlateCarree(), cmap=weatherbell_precip_colormap()[0], norm=weatherbell_precip_colormap()[1])
    # cf = ax.pcolormesh(time_ds['lon'], time_ds['lat'], time_ds['tp']*0.0393701, transform=ccrs.PlateCarree(), cmap=weatherbell_precip_colormap()[0], norm=weatherbell_precip_colormap()[1])
    ax.add_feature(cartopy.feature.STATES)
    # ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', linewidth=0.5)
    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.03)
    plt.savefig(f'figures/rt/total_tp_{t}.png')

    # fig = plt.figure(figsize=(10,10))
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # cf = ax.contourf(time_ds['lon'], time_ds['lat'], time_ds['tp']*0.0393701, transform=ccrs.PlateCarree(), levels=[0,0.01,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.5,3,3.5,4,5,6,7,8,9,10], cmap=weatherbell_precip_colormap()[0], norm=weatherbell_precip_colormap()[1])
    # ax.add_feature(cartopy.feature.STATES)
    # ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', linewidth=0.5)
    # cbar = plt.colorbar(cf, orientation='horizontal', pad=0.03)
    # # ax.set_extent([-109, -105, 37, 41])
    # plt.savefig(f'figures/rt/total_tp_{t}.png')


mean_final_total_ds = total_ds.isel(step=-1).mean(dim='number')
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection=ccrs.PlateCarree())
# cf = ax.pcolormesh(mean_final_total_ds['lon'], mean_final_total_ds['lat'], mean_final_total_ds['tp']*0.0393701, vmin=0, vmax=3)#, transform=ccrs.PlateCarree(), cmap=weatherbell_precip_colormap()[0], norm=weatherbell_precip_colormap()[1])
cf = ax.pcolormesh(mean_final_total_ds['lon'], mean_final_total_ds['lat'], mean_final_total_ds['tp']*0.0393701, transform=ccrs.PlateCarree(), cmap=weatherbell_precip_colormap()[0], norm=weatherbell_precip_colormap()[1])
ax.add_feature(cartopy.feature.STATES)
# ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', linewidth=0.5)
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.03)
plt.savefig(f'figures/rt/total_tp.png')

# fig = plt.figure(figsize=(10,10))
# ax = plt.axes(projection=ccrs.PlateCarree())
# cf = ax.contourf(mean_final_total_ds['lon'], mean_final_total_ds['lat'], mean_final_total_ds['tp']*0.0393701, transform=ccrs.PlateCarree(), levels=[0,0.01,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.5,3,3.5,4,5,6,7,8,9,10], cmap=weatherbell_precip_colormap()[0], norm=weatherbell_precip_colormap()[1])
# ax.add_feature(cartopy.feature.STATES)
# ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', linewidth=0.5)
# cbar = plt.colorbar(cf, orientation='horizontal', pad=0.03)
# # ax.set_extent([-109, -105, 37, 41])
# plt.savefig(f'figures/rt/total_tp.png')


base_point_data = total_ds.sel(lat=39.21340, lon=-106.92607, method='nearest').tp.values*0.0393701
plt.figure(figsize=(14, 6))
plt.boxplot(base_point_data, showfliers=False, whis=[5, 95])
plt.title('Accumulated Precipitation')
plt.xlabel('Timesteps')
plt.ylabel('Values')
plt.savefig('figures/rt/base_boxplot.png')

# 39.16827, -106.95216
top_point_data = total_ds.sel(lat=39.16827, lon=-106.95216, method='nearest').tp.values*0.0393701
plt.figure(figsize=(14, 6))
plt.boxplot(top_point_data, showfliers=False, whis=[5, 95])
plt.title('Accumulated Precipitation')
plt.xlabel('Timesteps')
plt.ylabel('Values')
plt.savefig('figures/rt/top_boxplot.png')

if rt_model == 'gfs' or rt_model == 'ecmwf':
    base_point_data = base_point_data[0]
    top_point_data = top_point_data[0]
    plt.figure(figsize=(14, 6))
    plt.plot(base_point_data, label='Base Point')
    plt.plot(top_point_data, label='Top Point')
    plt.title('Accumulated Precipitation')
    plt.xlabel('Timesteps')
    plt.ylabel('Values')
    plt.savefig('figures/rt/both_boxplot.png')
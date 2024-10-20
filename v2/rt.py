import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from tqdm import tqdm
import numpy as np
from metpy.plots import USCOUNTIES

from src.model import UNetWithAttention
from src.get_coordinates import get_coordinates
from src.realtime_ecmwf import realtime_ecmwf
from src.realtime_gfs import realtime_gfs
from src.realtime_gefs import realtime_gefs
from src.realtime_eps import realtime_eps



datestr, cycle = '20241018', '18'
frames = list(range(3, 91, 3))
ingest = False
tile = 89
model = 'ecmwf'


coarse_lats_pad, coarse_lons_pad, coarse_lats, coarse_lons, fine_lats, fine_lons = get_coordinates(tile)

if model == 'ecmwf':
    ds = realtime_ecmwf(datestr, cycle, frames, ingest)
    coarse_ds = ds.interp(lat=coarse_lats_pad, lon=coarse_lons_pad)
if model == 'gfs':
    ds = realtime_gfs(datestr, cycle, frames, ingest)
    coarse_ds = ds.interp(lat=coarse_lats_pad, lon=coarse_lons_pad)
if model == 'gefs':
    ds = realtime_gefs(datestr, cycle, frames, ingest)
    coarse_ds = ds.interp(lat=coarse_lats_pad, lon=coarse_lons_pad)
    coarse_ds = coarse_ds.transpose('step', 'number', 'lat', 'lon')
    for t in range(len(coarse_ds.valid_time)):
        if coarse_ds['valid_time'][t].dt.hour in [0, 6, 12, 18]:
            coarse_ds['tp'][t] = coarse_ds['tp'][t] - coarse_ds['tp'][t-1]
    coarse_ds = coarse_ds.transpose('number', 'step', 'lat', 'lon')
if model == 'eps':
    ds = realtime_eps(datestr, cycle, frames, ingest)
    coarse_ds = ds.interp(lat=coarse_lats_pad, lon=coarse_lons_pad)
    coarse_ds = coarse_ds.transpose('step', 'number', 'lat', 'lon')
    coarse_ds['cum_tp'] = coarse_ds['tp']*1000
    coarse_ds['tp'][0] = coarse_ds['cum_tp'][0]
    for t in range(1,len(coarse_ds.valid_time)):
        coarse_ds['tp'][t] = coarse_ds['cum_tp'][t]-coarse_ds['cum_tp'][t-1]
    coarse_ds = coarse_ds.transpose('number', 'step', 'lat', 'lon')
if model == 'super':
    import xarray as xr
    gefs_ds = realtime_gefs(datestr, cycle, frames, ingest)
    gefs_coarse_ds = gefs_ds.interp(lat=coarse_lats_pad, lon=coarse_lons_pad)
    gefs_coarse_ds = gefs_coarse_ds.transpose('step', 'number', 'lat', 'lon')
    for t in range(len(gefs_coarse_ds.valid_time)):
        if gefs_coarse_ds['valid_time'][t].dt.hour in [0, 6, 12, 18]:
            gefs_coarse_ds['tp'][t] = gefs_coarse_ds['tp'][t] - gefs_coarse_ds['tp'][t-1]
    gefs_coarse_ds = gefs_coarse_ds.transpose('number', 'step', 'lat', 'lon')

    eps_ds = realtime_eps(datestr, cycle, frames, ingest)
    eps_coarse_ds = eps_ds.interp(lat=coarse_lats_pad, lon=coarse_lons_pad)
    eps_coarse_ds = eps_coarse_ds.transpose('step', 'number', 'lat', 'lon')
    eps_coarse_ds['cum_tp'] = eps_coarse_ds['tp']*1000
    eps_coarse_ds['tp'][0] = eps_coarse_ds['cum_tp'][0]
    for t in range(1,len(eps_coarse_ds.valid_time)):
        eps_coarse_ds['tp'][t] = eps_coarse_ds['cum_tp'][t]-eps_coarse_ds['cum_tp'][t-1]
    eps_coarse_ds = eps_coarse_ds.transpose('number', 'step', 'lat', 'lon')

    coarse_ds = xr.concat([gefs_coarse_ds, eps_coarse_ds], dim='number')



tp = coarse_ds.tp.values

if len(tp.shape) == 3:
    tp = np.expand_dims(tp, axis=0)

tensor_tp = torch.tensor(tp, dtype=torch.float32).to('mps')

model = UNetWithAttention(1, 1, output_shape=(64,64)).to('mps')
checkpoint = torch.load(f'/Users/clamalo/documents/harpnet/v2/v2_checkpoints/best/{tile}_model.pt',)
# checkpoint = torch.load(f'/Users/clamalo/documents/harpnet/v2/{tile}_model.pt',)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():

    output = []
    for m in tqdm(range(0, tp.shape[0], 1)):
        input_arr = tensor_tp[m]

        m_output = model(input_arr).cpu().detach().numpy()
        output.append(np.expand_dims(m_output, axis=0))

output = np.concatenate(output, axis=0)


import matplotlib.colors as colors
def weatherbell_precip_colormap():
    cmap = colors.ListedColormap(['#ffffff', '#bdbfbd','#aba5a5', '#838383', '#6e6e6e', '#b5fbab', '#95f58b','#78f572', '#50f150','#1eb51e', '#0ea10e', '#1464d3', '#2883f1', '#50a5f5','#97d3fb', '#b5f1fb','#fffbab', '#ffe978', '#ffc13c', '#ffa100', '#ff6000','#ff3200', '#e11400','#c10000', '#a50000', '#870000', '#643c31', '#8d6558','#b58d83', '#c7a095','#f1ddd3', '#cecbdc'])#, '#aca0c7', '#9b89bd', '#725ca3','#695294', '#770077','#8d008d', '#b200b2', '#c400c4', '#db00db'])
    cmap.set_over(color='#aca0c7')
    bounds = [0,0.01,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.5,3,3.5,4,5,6,7,8,9,10]#,11,12,13,14,15,16,17,18,19,20])
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
    return cmap, norm, bounds



# for t in tqdm(range(tp.shape[1]), desc="Plotting Time Steps"):
#     fig, axes = plt.subplots(5, 6, figsize=(30, 25), subplot_kw={'projection': ccrs.PlateCarree()})
#     axes = axes.flatten()
#     # for member in range(output.shape[0]):
#     for member in range(30):
#         ax = axes[member]
#         cf = ax.pcolormesh(
#             fine_lons, fine_lats, 
#             output[member, t, :, :] * 0.0393701,  # Convert to inches if needed
#             transform=ccrs.PlateCarree(), 
#             cmap=weatherbell_precip_colormap()[0], 
#             norm=weatherbell_precip_colormap()[1])
#         ax.add_feature(cartopy.feature.STATES, linewidth=0.5)
#         ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', linewidth=0.5)
#         ax.set_title(f'Member {member + 1}', fontsize=12)
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
#     plt.savefig(f'figures/precipitation_t{int(t*3):03d}.png', dpi=300)
#     plt.close(fig)



total_tp = output.sum(axis=1)#.mean(axis=0)
total_input_tp = tp.sum(axis=1)#.mean(axis=0)



fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection=ccrs.PlateCarree())
cf = ax.pcolormesh(fine_lons, fine_lats, total_tp.mean(axis=0)*0.0393701, transform=ccrs.PlateCarree(), cmap=weatherbell_precip_colormap()[0], norm=weatherbell_precip_colormap()[1])
ax.add_feature(cartopy.feature.STATES)
ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', linewidth=0.5)
# ax.plot(-111.6400, 40.5881, 'ro', markersize=5, transform=ccrs.PlateCarree())
# ax.plot(-111.5354, 40.6461, 'ro', markersize=5, transform=ccrs.PlateCarree())
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.03)
plt.savefig(f'figures/total_tp.png')

fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection=ccrs.PlateCarree())
cf = ax.pcolormesh(coarse_lons, coarse_lats, total_input_tp.mean(axis=0)[1:-1, 1:-1]*0.0393701, transform=ccrs.PlateCarree(), cmap=weatherbell_precip_colormap()[0], norm=weatherbell_precip_colormap()[1])
ax.add_feature(cartopy.feature.STATES)
ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', linewidth=0.5)
# ax.plot(-111.6400, 40.5881, 'ro', markersize=5, transform=ccrs.PlateCarree())
# ax.plot(-111.5354, 40.6461, 'ro', markersize=5, transform=ccrs.PlateCarree())
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.03)
plt.savefig(f'figures/total_coarse_tp.png')
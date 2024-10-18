from src.get_coordinates import get_coordinates
from src.realtime_ecmwf import realtime_ecmwf
from src.realtime_gfs import realtime_gfs


datestr, cycle = '20241017', '12'
frames = list(range(3, 121, 3))
ingest = False
tile = 61

ds = realtime_ecmwf(datestr, cycle, frames, ingest)
# ds = realtime_gfs(datestr, cycle, frames, ingest)


import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from src.model import UNetWithAttention


coarse_lats_pad, coarse_lons_pad, coarse_lats, coarse_lons, fine_lats, fine_lons = get_coordinates(tile)
coarse_ds = ds.interp(lat=coarse_lats_pad, lon=coarse_lons_pad)

tp = coarse_ds.tp.values

tp = torch.tensor(coarse_ds.tp.values, dtype=torch.float32).to('mps')

model = UNetWithAttention(1, 1, output_shape=(64,64)).to('mps')
checkpoint = torch.load('/Volumes/T9/v2_checkpoints/best/61_model.pt',)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
with torch.no_grad():
    output = model(tp).cpu().detach().numpy()


# for t in range(tp.shape[0]):
#     fig = plt.figure(figsize=(10,10))
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ax.pcolormesh(coarse_lons_pad[1:-1], coarse_lats_pad[1:-1], tp.cpu().detach().numpy()[t][1:-1, 1:-1]*0.0393701, transform=ccrs.PlateCarree(), vmin=0, vmax=0.5)
#     ax.add_feature(cartopy.feature.STATES)
#     plt.savefig(f'figures/tp_{t}.png')
#     plt.close()
    
#     fig = plt.figure(figsize=(10,10))
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ax.pcolormesh(fine_lons, fine_lats, output[t]*0.0393701, transform=ccrs.PlateCarree(), vmin=0, vmax=0.5)
#     ax.add_feature(cartopy.feature.STATES)
#     plt.savefig(f'figures/out_tp_{t}.png')
#     plt.close()


total_tp = output.sum(axis=0)
print(total_tp.shape)


import matplotlib.colors as colors
def weatherbell_precip_colormap():
    cmap = colors.ListedColormap(['#ffffff', '#bdbfbd','#aba5a5', '#838383', '#6e6e6e', '#b5fbab', '#95f58b','#78f572', '#50f150','#1eb51e', '#0ea10e', '#1464d3', '#2883f1', '#50a5f5','#97d3fb', '#b5f1fb','#fffbab', '#ffe978', '#ffc13c', '#ffa100', '#ff6000','#ff3200', '#e11400','#c10000', '#a50000', '#870000', '#643c31', '#8d6558','#b58d83', '#c7a095','#f1ddd3', '#cecbdc'])#, '#aca0c7', '#9b89bd', '#725ca3','#695294', '#770077','#8d008d', '#b200b2', '#c400c4', '#db00db'])
    cmap.set_over(color='#aca0c7')
    bounds = [0,0.01,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.5,3,3.5,4,5,6,7,8,9,10]#,11,12,13,14,15,16,17,18,19,20])
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))
    return cmap, norm, bounds

fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.pcolormesh(fine_lons, fine_lats, total_tp*0.0393701, transform=ccrs.PlateCarree(), cmap=weatherbell_precip_colormap()[0], norm=weatherbell_precip_colormap()[1])
ax.add_feature(cartopy.feature.STATES)
from metpy.plots import USCOUNTIES
ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', linewidth=0.5)
# add a point at alta ski area
ax.plot(-111.6400, 40.5881, 'ro', markersize=5, transform=ccrs.PlateCarree())
# add a point at park city ski area
ax.plot(-111.5354, 40.6461, 'ro', markersize=5, transform=ccrs.PlateCarree())
plt.savefig('figures/total_tp.png')

# closest to alta
import numpy as np
closest_lat_idx = np.abs(fine_lats - 40.5881).argmin()
closest_lon_idx = np.abs(fine_lons - -111.6400).argmin()
print(total_tp[closest_lat_idx, closest_lon_idx]*0.0393701)

closest_lat_idx = np.abs(fine_lats - 40.6461).argmin()
closest_lon_idx = np.abs(fine_lons - -111.5354).argmin()
print(total_tp[closest_lat_idx, closest_lon_idx]*0.0393701)
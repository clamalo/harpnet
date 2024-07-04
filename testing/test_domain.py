import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from metpy.plots import USCOUNTIES
import time
import matplotlib.colors as colors
def weatherbell_precip_colormap():
    cmap = colors.ListedColormap(['#ffffff', '#bdbfbd','#aba5a5', '#838383', '#6e6e6e', '#b5fbab', '#95f58b','#78f572', '#50f150','#1eb51e', '#0ea10e', '#1464d3', '#2883f1', '#50a5f5','#97d3fb', '#b5f1fb','#fffbab', '#ffe978', '#ffc13c', '#ffa100', '#ff6000','#ff3200', '#e11400','#c10000', '#a50000', '#870000', '#643c31', '#8d6558','#b58d83', '#c7a095','#f1ddd3', '#cecbdc'])#, '#aca0c7', '#9b89bd', '#725ca3','#695294', '#770077','#8d008d', '#b200b2', '#c400c4', '#db00db'])
    return cmap
colormap = weatherbell_precip_colormap()
colormap.set_over(color='#aca0c7')
bounds = [0,0.01,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.5,3,3.5,4,5,6,7,8,9,10]#,11,12,13,14,15,16,17,18,19,20])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds))

max_lat, min_lat, max_lon, min_lon = 43.45, 37.9, -108.9, -114.45

reference_ds = xr.load_dataset('/Users/clamalo/documents/projects/conus404/data/reference_ds.grib2', engine='cfgrib')
reference_ds = reference_ds.assign_coords(longitude=(((reference_ds.longitude + 180) % 360) - 180)).sortby('longitude')
reference_ds = reference_ds.sel(latitude=slice(max_lat+0.5, min_lat-0.5), longitude=slice(min_lon-0.5, max_lon+0.5))

# open every file in /Volumes/T9/summed/ and concatenate them
datasets = []
# for month in [10, 11]:
for month in [11]:
    ds = xr.open_dataset(f'/Volumes/T9/monthly/1979-{month:02d}.nc')
    datasets.append(ds)
ds = xr.concat(datasets, dim='time')
ds = ds.sum(dim='time')
ds['tp'] = ds['tp'] * 0.0393701

coarse_ds = ds.interp(lat=reference_ds.latitude.values, lon=reference_ds.longitude.values)

ds = ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

print(len(coarse_ds.lat), len(coarse_ds.lon))
print(len(ds.lat), len(ds.lon))

#plot ds and coarse_ds side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
axs[0].pcolormesh(coarse_ds['lon'], coarse_ds['lat'], coarse_ds['tp'], transform=ccrs.PlateCarree(), cmap=colormap, norm=norm)
axs[0].add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', linewidth=0.5)
axs[0].set_title('Coarse')
axs[1].pcolormesh(ds['lon'], ds['lat'], ds['tp'], transform=ccrs.PlateCarree(), cmap=colormap, norm=norm)
axs[1].add_feature(USCOUNTIES.with_scale('5m'), edgecolor='black', linewidth=0.5)
axs[1].set_title('Fine')

axs[0].set_extent([min_lon-0.5, max_lon+0.5, min_lat-0.5, max_lat+0.5])
axs[1].set_extent([min_lon-0.5, max_lon+0.5, min_lat-0.5, max_lat+0.5])
 
plt.savefig(f'comparison.png', dpi=300, bbox_inches='tight')
"""
Plot and save a figure showing the defined grid tiles.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from src.get_coordinates import tiles
from src.constants import FIGURES_DIR

def plot_tiles():
    """
    Plot the tile grid and save as an image.
    """
    grid_tiles = tiles()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.STATES)

    for tile in grid_tiles:
        lat_min, lat_max = grid_tiles[tile][0]
        lon_min, lon_max = grid_tiles[tile][1]
        ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min],
                [lat_min, lat_min, lat_max, lat_max, lat_min],
                color='red', linewidth=2, transform=ccrs.PlateCarree())
        ax.text((lon_min + lon_max) / 2, (lat_min + lat_max) / 2, str(tile),
                horizontalalignment='center', verticalalignment='center', color='black', fontsize=10, 
                transform=ccrs.PlateCarree())
    plt.title('HARPNET Tiles')
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'grid_tiles.png'))
    plt.close(fig)
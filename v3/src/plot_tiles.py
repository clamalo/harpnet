"""
Plot and save a figure showing the defined grid tiles.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from src.create_grid_tiles import create_grid_tiles
from src.constants import FIGURES_DIR

def plot_tiles():
    """
    Plot the tile grid and save as an image.
    """
    grid_tiles = create_grid_tiles()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.STATES)

    for tile in grid_tiles:
        tile_min_lat, tile_max_lat, tile_min_lon, tile_max_lon = grid_tiles[tile]
        ax.plot([tile_min_lon, tile_max_lon, tile_max_lon, tile_min_lon, tile_min_lon],
                [tile_min_lat, tile_min_lat, tile_max_lat, tile_max_lat, tile_min_lat],
                color='red', linewidth=2, transform=ccrs.PlateCarree())
        ax.text((tile_min_lon + tile_max_lon) / 2, (tile_min_lat + tile_max_lat) / 2, str(tile),
                horizontalalignment='center', verticalalignment='center', color='black', fontsize=10, 
                transform=ccrs.PlateCarree())
    plt.title('HARPNET Tiles')
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'grid_tiles.png'))
    plt.close(fig)
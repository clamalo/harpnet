"""
Plots the defined tiles over a map using Cartopy and saves the figure.
Each tile is drawn as a rectangular outline with its tile index labeled.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from src.tiles import get_all_tiles
from src.constants import FIGURES_DIR

def plot_tiles():
    """
    Creates and saves a map showing the tile grid boundaries and indices.
    Tiles are outlined in red, and each tile index is labeled in the center.
    """
    grid_tiles = get_all_tiles()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.STATES)

    # --- DRAW TILE RECTANGLES AND LABELS ---
    for tile in grid_tiles:
        lat_min, lat_max, lon_min, lon_max = grid_tiles[tile]
        ax.plot(
            [lon_min, lon_max, lon_max, lon_min, lon_min],
            [lat_min, lat_min, lat_max, lat_max, lat_min],
            color='red', linewidth=2, transform=ccrs.PlateCarree()
        )
        ax.text(
            (lon_min + lon_max) / 2,
            (lat_min + lat_max) / 2,
            str(tile),
            horizontalalignment='center',
            verticalalignment='center',
            color='black',
            fontsize=10,
            transform=ccrs.PlateCarree()
        )

    plt.title('HARPNET Tiles')
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'grid_tiles.png'))
    plt.close(fig)
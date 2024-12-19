"""
Plot and save a figure showing the defined grid tiles.
"""

import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from src.utils.create_grid_tiles import create_grid_tiles
from src.config import Config
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def plot_tiles() -> None:
    """
    Plot the tile grid and save as an image.
    """
    grid_tiles = create_grid_tiles()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.STATES)

    for tile, box in grid_tiles.items():
        tile_min_lat, tile_max_lat, tile_min_lon, tile_max_lon = box
        ax.plot(
            [tile_min_lon, tile_max_lon, tile_max_lon, tile_min_lon, tile_min_lon],
            [tile_min_lat, tile_min_lat, tile_max_lat, tile_max_lat, tile_min_lat],
            color='red', linewidth=2, transform=ccrs.PlateCarree()
        )
        ax.text(
            (tile_min_lon + tile_max_lon) / 2,
            (tile_min_lat + tile_max_lat) / 2,
            str(tile),
            horizontalalignment='center', verticalalignment='center', color='black', fontsize=10,
            transform=ccrs.PlateCarree()
        )

    plt.title('HARPNET Tiles')
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(Config.FIGURES_DIR, 'grid_tiles.png')
    plt.savefig(fig_path)
    plt.close(fig)
    logger.info(f"Saved grid tiles figure to {fig_path}")
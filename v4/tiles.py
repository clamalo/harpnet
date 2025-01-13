from config import (
    PRIMARY_MIN_LAT, PRIMARY_MAX_LAT,
    PRIMARY_MIN_LON, PRIMARY_MAX_LON,
    SECONDARY_MIN_LAT, SECONDARY_MAX_LAT,
    SECONDARY_MIN_LON, SECONDARY_MAX_LON,
    TILE_SIZE, FINE_RESOLUTION, PADDING,
    COARSE_RESOLUTION, SECONDARY_TILES
)

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle

def get_tile_dict():
    """
    Generate tile boundaries for primary coordinates and, optionally, secondary coordinates.
    
    If SECONDARY_TILES is True, this function will also generate tile boundaries for the
    secondary region. Otherwise, it will only generate tiles for the primary region.
    
    Returns:
        dict:
            A dictionary keyed by an integer tile ID. The value for each key is a tuple:
            (min_lat, max_lat, min_lon, max_lon, tile_type).
            Where:
                - min_lat and max_lat are the latitude boundaries,
                - min_lon and max_lon are the longitude boundaries,
                - tile_type is either 'primary' or 'secondary'.
    """
    def build_tiles(lat_min, lat_max, lon_min, lon_max, deg, t_type):
        """
        Build a list of tiles for a given bounding box and tile size in degrees (deg).
        Each tile is defined only if it fits entirely within the specified bounding box.
        
        Args:
            lat_min (float): Minimum latitude of the bounding box.
            lat_max (float): Maximum latitude of the bounding box.
            lon_min (float): Minimum longitude of the bounding box.
            lon_max (float): Maximum longitude of the bounding box.
            deg (float): Tile size in degrees (TILE_SIZE * FINE_RESOLUTION).
            t_type (str): Either 'primary' or 'secondary'.
        
        Returns:
            list:
                A list of tuples, where each tuple is (tile_min_lat, tile_max_lat,
                tile_min_lon, tile_max_lon, tile_type).
        """
        tiles = []
        for lat in np.arange(lat_min, lat_max, deg):
            if lat + deg > lat_max:
                break
            for lon in np.arange(lon_min, lon_max, deg):
                if lon + deg > lon_max:
                    break
                tiles.append((lat, lat + deg, lon, lon + deg, t_type))
        return tiles

    # Define the tile size in degrees based on TILE_SIZE and FINE_RESOLUTION.
    tile_deg = TILE_SIZE * FINE_RESOLUTION

    # Build primary tiles.
    primary_tiles = build_tiles(
        lat_min=PRIMARY_MIN_LAT,
        lat_max=PRIMARY_MAX_LAT,
        lon_min=PRIMARY_MIN_LON,
        lon_max=PRIMARY_MAX_LON,
        deg=tile_deg,
        t_type='primary'
    )

    # Optionally build secondary tiles if SECONDARY_TILES is True.
    if SECONDARY_TILES:
        secondary_tiles = build_tiles(
            lat_min=SECONDARY_MIN_LAT,
            lat_max=SECONDARY_MAX_LAT,
            lon_min=SECONDARY_MIN_LON,
            lon_max=SECONDARY_MAX_LON,
            deg=tile_deg,
            t_type='secondary'
        )
    else:
        secondary_tiles = []

    # Combine all tiles (primary and optional secondary) and assign each one a unique integer ID.
    all_tiles = primary_tiles + secondary_tiles
    tile_dict = {i: tile for i, tile in enumerate(all_tiles)}

    return tile_dict


def tile_coordinates(tile_id):
    """
    Retrieve and return the coordinate arrays for the specified tile.
    
    The function returns four 1D arrays: lat_coarse, lon_coarse, lat_fine, lon_fine.
    The coarse grids include a PADDING on each side (top/bottom or left/right), while
    the fine grids do not include any padding. The arrays are generated using numpy.arange
    and do not include their upper boundary.

    Args:
        tile_id (int):
            The key of the tile in the dictionary returned by get_tile_dict().

    Returns:
        tuple of numpy.ndarray:
            (lat_coarse, lon_coarse, lat_fine, lon_fine)
            
            - lat_coarse, lon_coarse: Coarse-resolution coordinates with PADDING on all sides.
            - lat_fine, lon_fine: Fine-resolution coordinates without any padding.
    """
    tile_dict = get_tile_dict()
    min_lat, max_lat, min_lon, max_lon, _ = tile_dict[tile_id]

    # Generate coarse-resolution arrays with padding, excluding the upper boundary.
    lat_coarse = np.arange(
        start=min_lat - PADDING,
        stop=max_lat + PADDING,
        step=COARSE_RESOLUTION
    )
    lon_coarse = np.arange(
        start=min_lon - PADDING,
        stop=max_lon + PADDING,
        step=COARSE_RESOLUTION
    )

    # Generate fine-resolution arrays without padding, excluding the upper boundary.
    lat_fine = np.arange(
        start=min_lat,
        stop=max_lat,
        step=FINE_RESOLUTION
    )
    lon_fine = np.arange(
        start=min_lon,
        stop=max_lon,
        step=FINE_RESOLUTION
    )

    return lat_coarse, lon_coarse, lat_fine, lon_fine



if __name__ == "__main__":
    # Create a plot showing the tile boundaries.
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-130, -100, 30, 55], crs=ccrs.PlateCarree())

    # Add state borders and coastlines for context.
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plot each tile as a rectangle with a label.
    for tile_id in get_tile_dict():
        min_lat, max_lat, min_lon, max_lon, tile_type = get_tile_dict()[tile_id]
        ax.add_patch(Rectangle(
            xy=(min_lon, min_lat),
            width=max_lon - min_lon,
            height=max_lat - min_lat,
            edgecolor='black',
            facecolor='none',
            linestyle='--' if tile_type == 'secondary' else '-',
            label=tile_type
        ))

    # Add a legend for the tile types.
    ax.legend(loc='upper right', title='Tile Type')

    plt.title("Tile Boundaries")
    plt.show()
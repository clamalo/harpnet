"""
Functions for creating tile definitions over a geographic region.
"""

from src.config import Config

def create_grid_tiles() -> dict:
    """
    Create a dictionary of tile domains. Each tile is defined by a lat/lon bounding box.
    Tiles are created based on a fixed tile size of (16 / SCALE_FACTOR).

    Returns:
        dict: A dictionary where keys are tile indices and values are [lat_min, lat_max, lon_min, lon_max].
    """
    tile_size_degrees = int(16 / Config.SCALE_FACTOR)
    grid_domains = {}
    total_domains = 0
    for lat in range(Config.MIN_LAT, Config.MAX_LAT, tile_size_degrees):
        for lon in range(Config.MIN_LON, Config.MAX_LON, tile_size_degrees):
            grid_domains[total_domains] = [
                lat, lat + tile_size_degrees, lon, lon + tile_size_degrees
            ]
            total_domains += 1
    return grid_domains
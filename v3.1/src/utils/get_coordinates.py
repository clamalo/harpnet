"""
Utilities to get coordinates (lat/lon) for a given tile.
"""

import numpy as np
from src.utils.create_grid_tiles import create_grid_tiles
from src.config import Config

def get_coordinates(tile: int):
    """
    Given a tile index, return coarse and fine lat/lon coordinates.

    Args:
        tile (int): Tile index.

    Returns:
        tuple: coarse_lats_pad, coarse_lons_pad, coarse_lats, coarse_lons, fine_lats, fine_lons
    """

    def scale_coordinates(coords, scale_factor):
        """
        Convert coarse coordinates into finer resolution coordinates.
        """
        resolution = coords[1] - coords[0]
        fine_resolution = resolution / scale_factor
        x_fine = []
        for center in coords:
            start = center - (resolution / 2) + (fine_resolution / 2)
            fine_points = [start + i * fine_resolution for i in range(scale_factor)]
            x_fine.extend(fine_points)
        return np.array(x_fine)

    grid_tiles = create_grid_tiles()
    tile_min_lat, tile_max_lat, tile_min_lon, tile_max_lon = grid_tiles[tile]

    resolution = 0.25
    coarse_lats_pad = np.arange(tile_min_lat - resolution, tile_max_lat + resolution, resolution)
    coarse_lons_pad = np.arange(tile_min_lon - resolution, tile_max_lon + resolution, resolution)
    coarse_lats = np.arange(tile_min_lat, tile_max_lat, resolution)
    coarse_lons = np.arange(tile_min_lon, tile_max_lon, resolution)
    fine_lats = scale_coordinates(coarse_lats, Config.SCALE_FACTOR)
    fine_lons = scale_coordinates(coarse_lons, Config.SCALE_FACTOR)

    return coarse_lats_pad, coarse_lons_pad, coarse_lats, coarse_lons, fine_lats, fine_lons
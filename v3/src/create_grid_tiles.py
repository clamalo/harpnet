"""
Functions for creating tile definitions over a geographic region using the updated tile logic.
"""

from src.constants import (MIN_LAT, MAX_LAT, MIN_LON, MAX_LON,
                           TILE_SIZE, FINE_RESOLUTION)

def create_grid_tiles():
    """
    Create a dictionary of tile domains. Each tile is defined by a lat/lon bounding box.
    Tiles cover a certain geographic area based on TILE_SIZE and FINE_RESOLUTION.

    TILE_SIZE x TILE_SIZE pixels at FINE_RESOLUTION degrees per pixel means each tile covers
    TILE_SIZE * FINE_RESOLUTION degrees in both latitude and longitude.

    Returns:
        A dictionary where each key is a tile index and each value is
        [lat_min, lat_max, lon_min, lon_max].
    """
    tile_size_degrees = TILE_SIZE * FINE_RESOLUTION
    tiles_dict = {}
    tile_counter = 0
    
    current_lat = MIN_LAT
    while current_lat + tile_size_degrees <= MAX_LAT:
        lat_upper = current_lat + tile_size_degrees
        
        current_lon = MIN_LON
        while current_lon + tile_size_degrees <= MAX_LON:
            lon_upper = current_lon + tile_size_degrees
            
            tiles_dict[tile_counter] = [current_lat, lat_upper, current_lon, lon_upper]
            tile_counter += 1
            
            current_lon += tile_size_degrees
            
        current_lat += tile_size_degrees
    
    return tiles_dict
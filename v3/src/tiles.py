import numpy as np
from src.constants import MIN_LAT, MIN_LON, MAX_LAT, MAX_LON, TILE_SIZE, COARSE_RESOLUTION, FINE_RESOLUTION, PADDING

def get_all_tiles():
    """
    Generate a dictionary of all tiles over the specified lat/lon domain.
    Each tile covers TILE_SIZE * FINE_RESOLUTION degrees in both latitude and longitude.
    Returns:
        A dictionary {tile_index: (lat_min, lat_max, lon_min, lon_max)}.
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
            tiles_dict[tile_counter] = (current_lat, lat_upper, current_lon, lon_upper)
            tile_counter += 1
            
            current_lon += tile_size_degrees
            
        current_lat += tile_size_degrees
    
    return tiles_dict

def tile_coordinates(tile_index):
    """
    Given a tile index, return coarse and fine coordinate arrays.
    The tile bounding box is obtained from get_all_tiles().
    Returns:
        coarse_latitudes, coarse_longitudes, fine_latitudes, fine_longitudes
    """
    lat_min, lat_max, lon_min, lon_max = get_all_tiles()[tile_index]

    # Fine-resolution arrays
    fine_latitudes = np.arange(lat_min, lat_max, FINE_RESOLUTION)
    fine_longitudes = np.arange(lon_min, lon_max, FINE_RESOLUTION)

    # Apply padding to coarse arrays if PADDING is a float/int
    if isinstance(PADDING, (float, int)):
        coarse_min_lat = lat_min - PADDING
        coarse_max_lat = lat_max + PADDING
        coarse_min_lon = lon_min - PADDING
        coarse_max_lon = lon_max + PADDING
    else:
        coarse_min_lat = lat_min
        coarse_max_lat = lat_max
        coarse_min_lon = lon_min
        coarse_max_lon = lon_max

    # Coarse-resolution arrays
    coarse_latitudes = np.arange(coarse_min_lat, coarse_max_lat, COARSE_RESOLUTION)
    coarse_longitudes = np.arange(coarse_min_lon, coarse_max_lon, COARSE_RESOLUTION)

    return coarse_latitudes, coarse_longitudes, fine_latitudes, fine_longitudes
import numpy as np
from src.constants import MIN_LAT, MIN_LON, MAX_LAT, MAX_LON, TILE_SIZE, COARSE_RESOLUTION, FINE_RESOLUTION, PADDING

def tiles():
    # Determine how many degrees each tile covers
    tile_size_degrees = TILE_SIZE * FINE_RESOLUTION
    
    tiles_dict = {}
    tile_counter = 0
    
    current_lat = MIN_LAT
    while current_lat + tile_size_degrees <= MAX_LAT:
        lat_upper = current_lat + tile_size_degrees
        
        current_lon = MIN_LON
        while current_lon + tile_size_degrees <= MAX_LON:
            lon_upper = current_lon + tile_size_degrees
            
            # Store the full tile
            tiles_dict[tile_counter] = [[current_lat, lat_upper],
                                        [current_lon, lon_upper]]
            tile_counter += 1
            
            current_lon += tile_size_degrees
            
        current_lat += tile_size_degrees
    
    return tiles_dict


def tile_coordinates(tile):
    bounds = tiles()[tile]

    min_lat, max_lat = bounds[0]
    min_lon, max_lon = bounds[1]

    # Fine-resolution arrays
    fine_latitudes = np.arange(min_lat, max_lat, FINE_RESOLUTION)
    fine_longitudes = np.arange(min_lon, max_lon, FINE_RESOLUTION)

    # Apply padding to coarse arrays if PADDING is a float/int
    if isinstance(PADDING, (float, int)):
        coarse_min_lat = min_lat - PADDING
        coarse_max_lat = max_lat + PADDING
        coarse_min_lon = min_lon - PADDING
        coarse_max_lon = max_lon + PADDING
    else:
        coarse_min_lat = min_lat
        coarse_max_lat = max_lat
        coarse_min_lon = min_lon
        coarse_max_lon = max_lon

    # Coarse-resolution arrays
    coarse_latitudes = np.arange(coarse_min_lat, coarse_max_lat, COARSE_RESOLUTION)
    coarse_longitudes = np.arange(coarse_min_lon, coarse_max_lon, COARSE_RESOLUTION)

    return coarse_latitudes, coarse_longitudes, fine_latitudes, fine_longitudes
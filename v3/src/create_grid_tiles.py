from src.constants import SCALE_FACTOR, MIN_LAT, MIN_LON, MAX_LAT, MAX_LON

def create_grid_tiles():
    tile_size_degrees = int(16/SCALE_FACTOR)
    grid_domains = {}
    total_domains = 0
    for lat in range(MIN_LAT, MAX_LAT, tile_size_degrees):
        for lon in range(MIN_LON, MAX_LON, tile_size_degrees):
            grid_domains[total_domains] = [lat, lat + tile_size_degrees, lon, lon + tile_size_degrees]
            total_domains += 1

    return grid_domains
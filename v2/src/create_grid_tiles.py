import src.constants as constants

def create_grid_tiles():
    tile_size_degrees = int(16/constants.scale_factor)
    min_lat, min_lon = constants.min_lat, constants.min_lon
    max_lat, max_lon = constants.max_lat, constants.max_lon
    grid_domains = {}
    total_domains = 0
    for lat in range(min_lat, max_lat, tile_size_degrees):
        for lon in range(min_lon, max_lon, tile_size_degrees):
            grid_domains[total_domains] = [lat, lat + tile_size_degrees, lon, lon + tile_size_degrees]
            total_domains += 1

    return grid_domains
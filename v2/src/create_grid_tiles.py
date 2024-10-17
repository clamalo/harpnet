import src.constants as constants

def create_grid_tiles():
    min_lat, min_lon = constants.min_lat, constants.min_lon
    max_lat, max_lon = constants.max_lat, constants.max_lon
    grid_domains = {}
    total_domains = 0
    for lat in range(min_lat, max_lat, 4):
        for lon in range(min_lon, max_lon, 4):
            grid_domains[total_domains] = [lat, lat + 4, lon, lon + 4]
            total_domains += 1

    return grid_domains
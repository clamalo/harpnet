import constants

def create_grid_domains():
    start_lat, start_lon = constants.start_lat, constants.start_lon
    end_lat, end_lon = constants.end_lat, constants.end_lon
    grid_domains = {}
    total_domains = 0
    for lat in range(start_lat, end_lat, 4):
        for lon in range(start_lon, end_lon, 4):
            grid_domains[total_domains] = [lat, lat + 4, lon, lon + 4]
            total_domains += 1

    return grid_domains
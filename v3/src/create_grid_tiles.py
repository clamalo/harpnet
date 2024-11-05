from src.constants import SCALE_FACTOR, OUTER_GRID, INNER_GRID

def create_grid_tiles():
    tile_size_degrees = int(16/SCALE_FACTOR)

    grid_domains = {}
    total_domains = 0
    MIN_LAT, MIN_LON, MAX_LAT, MAX_LON = OUTER_GRID['MIN_LAT'], OUTER_GRID['MIN_LON'], OUTER_GRID['MAX_LAT'], OUTER_GRID['MAX_LON']
    for lat in range(MIN_LAT, MAX_LAT+1, tile_size_degrees):
        for lon in range(MIN_LON, MAX_LON+1, tile_size_degrees):
            grid_domains[total_domains] = [lat, lat + tile_size_degrees, lon, lon + tile_size_degrees]
            total_domains += 1
    MIN_LAT, MIN_LON, MAX_LAT, MAX_LON = INNER_GRID['MIN_LAT'], INNER_GRID['MIN_LON'], INNER_GRID['MAX_LAT'], INNER_GRID['MAX_LON']
    for lat in range(MIN_LAT, MAX_LAT+1, tile_size_degrees):
        for lon in range(MIN_LON, MAX_LON+1, tile_size_degrees):
            grid_domains[total_domains] = [lat, lat + tile_size_degrees, lon, lon + tile_size_degrees]
            total_domains += 1

    return grid_domains
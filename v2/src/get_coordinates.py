from src.create_grid_tiles import create_grid_tiles
import src.constants as constants
import numpy as np

def get_coordinates(tile):
    resolution = 0.25
    grid_tiles = create_grid_tiles()
    def scale_coordinates(x, scale_factor):
        resolution = x[1] - x[0]
        fine_resolution = resolution / scale_factor
        x_fine = []
        for center in x:
            start = center - (resolution / 2) + (fine_resolution / 2)
            fine_points = [start + i * fine_resolution for i in range(scale_factor)]
            x_fine.extend(fine_points)
        return np.array(x_fine)
    tile_min_lat, tile_max_lat, tile_min_lon, tile_max_lon = grid_tiles[tile]
    coarse_lats_pad = np.arange(tile_min_lat-resolution, tile_max_lat+resolution, resolution)
    coarse_lons_pad = np.arange(tile_min_lon-resolution, tile_max_lon+resolution, resolution)
    coarse_lats = np.arange(tile_min_lat, tile_max_lat, resolution)
    coarse_lons = np.arange(tile_min_lon, tile_max_lon, resolution)
    fine_lats = scale_coordinates(coarse_lats, constants.scale_factor)
    fine_lons = scale_coordinates(coarse_lons, constants.scale_factor)

    return coarse_lats_pad, coarse_lons_pad, coarse_lats, coarse_lons, fine_lats, fine_lons
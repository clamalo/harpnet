from src.create_grid_domains import create_grid_domains
import numpy as np

def get_coordinates(domain):
    resolution = 0.25
    grid_domains = create_grid_domains()
    def scale_coordinates(x, scale_factor):
        resolution = x[1] - x[0]
        fine_resolution = resolution / scale_factor
        x_fine = []
        for center in x:
            start = center - (resolution / 2) + (fine_resolution / 2)
            fine_points = [start + i * fine_resolution for i in range(scale_factor)]
            x_fine.extend(fine_points)
        return np.array(x_fine)
    min_lat, max_lat, min_lon, max_lon = grid_domains[domain]
    coarse_lats_pad = np.arange(min_lat-resolution, max_lat+resolution, resolution)
    coarse_lons_pad = np.arange(min_lon-resolution, max_lon+resolution, resolution)
    coarse_lats = np.arange(min_lat, max_lat, resolution)
    coarse_lons = np.arange(min_lon, max_lon, resolution)
    fine_lats = scale_coordinates(coarse_lats, 4)
    fine_lons = scale_coordinates(coarse_lons, 4)

    return coarse_lats_pad, coarse_lons_pad, coarse_lats, coarse_lons, fine_lats, fine_lons
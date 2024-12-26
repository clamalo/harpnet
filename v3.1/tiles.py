from constants import PRIMARY_MIN_LAT, PRIMARY_MAX_LAT, PRIMARY_MIN_LON, PRIMARY_MAX_LON, SECONDARY_MIN_LAT, SECONDARY_MAX_LAT, TILE_SIZE, COARSE_RESOLUTION, FINE_RESOLUTION, PADDING

def get_tile_dict():
    # There are two sets of coordinate ranges: primary and secondary
    # Primary is the "base" set of tiles, secondary is laid on top and staggered by half a tile width
    # The purpose of this function is to use the variables in constants.py to generate a dictionary of tile coordinates
    # Each tile will be a number, and the value of each tile will be a tuple with the tile's min and max lat and lon

    # Start from the furthest south and west coordinates. Go across to the furthest south and east coordinates, then go up a row and repeat.

    # Do the primary set of tiles first, then the secondary set.

    # Finally, plot all the tiles on a map to make sure they're correct. Use matplotlib and cartopy with state lines. Primary tiles plotted first in red, then secondary tiles plotted on top in blue.
    pass
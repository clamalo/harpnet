from src.plot_tiles import plot_tiles
from src.setup import setup
from src.xr_to_np import xr_to_np
from src.generate_dataloaders import generate_dataloaders
from src.train_test import train_test
from src.ensemble import ensemble
from src.cleanup import cleanup

# Variables
start_month = (1979, 10)
end_month = (1980, 9)
train_test_ratio = 0.2
start_epoch, end_epoch = 20, 25
zip_setting = 'save'    # False, 'load', or 'save'
max_ensemble_size = 8

# We now define multiple tiles to be processed at once
tiles = [0, 1, 2]

# Setup directories for all tiles
for tile in tiles:
    setup(tile)

# Preprocess multiple tiles at once
xr_to_np(tiles, start_month, end_month, zip_setting)

if zip_setting == 'save':
    # If we are saving zips, we stop here
    exit()

# Generate dataloaders from multiple tiles combined
train_dataloader, test_dataloader = generate_dataloaders(tiles, start_month, end_month, train_test_ratio)

# Note: The training and testing functions currently assume one tile at a time. 
# If you want to train a single model over multiple tiles combined, you can proceed as is.
# If you need separate models per tile, loop over tiles again. 
# For demonstration, we'll just pick one tile to run train/test on.
tile = tiles[0]

train_test(tile, train_dataloader, test_dataloader, start_epoch, end_epoch)
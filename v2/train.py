from src.plot_tiles import plot_tiles
from src.setup import setup
from src.xr_to_np import xr_to_np
from src.generate_dataloaders import generate_dataloaders
from src.train_test import train_test
from src.ensemble import ensemble

# Variables
start_month = (1979, 10)
end_month = (2022, 9)
train_test_ratio = 0.2
start_epoch, end_epoch = 0, 20
max_ensemble_size = 8

# plot_tiles()

tiles = [7]

for tile in tiles:

    setup(tile)

    xr_to_np(tile, start_month, end_month)

    train_dataloader, test_dataloader = generate_dataloaders(tile, start_month, end_month, train_test_ratio)

    train_test(tile, train_dataloader, test_dataloader, start_epoch, end_epoch)

    ensemble(tile, start_month, end_month, train_test_ratio, max_ensemble_size)
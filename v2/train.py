from src.plot_tiles import plot_tiles
from src.setup import setup
from src.xr_to_np import xr_to_np
from src.generate_dataloaders import generate_dataloaders
from src.train_test import train_test
from src.ensemble import ensemble

# Variables
start_month = (1984, 10)
end_month = (1984, 10)
train_test_ratio = 0.2
max_ensemble_size = 6

# plot_tiles()

# tiles = list(range(15, 16))
tiles = [52]#,53,62,63]

for tile in tiles:

    setup(tile)

    xr_to_np(tile, start_month, end_month)
    
    quit()

    train_dataloader, test_dataloader = generate_dataloaders(tile, start_month, end_month, train_test_ratio)

    train_test(tile, train_dataloader, test_dataloader, epochs=20)

    ensemble(tile, start_month, end_month, train_test_ratio, max_ensemble_size)
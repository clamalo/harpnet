from src.plot_tiles import plot_tiles
from src.setup import setup
from src.xr_to_np import xr_to_np
from src.generate_dataloaders import generate_dataloaders
from src.train_test import train_test
from src.ensemble import ensemble
from src.cleanup import cleanup

# Variables
start_month = (1979, 10)
end_month = (1979, 10)
train_test_ratio = 0.2
start_epoch, end_epoch = 0, 20
zip_setting = False    # False, 'load', or 'save'
max_ensemble_size = 8

# plot_tiles()

setup()

xr_to_np(start_month, end_month)

if zip_setting == 'save':
    quit()

train_dataloader, test_dataloader = generate_dataloaders(start_month, end_month, train_test_ratio)

print(len(train_dataloader), len(test_dataloader))

train_test(train_dataloader, test_dataloader, start_epoch, end_epoch)

quit()

ensemble(tile, start_month, end_month, train_test_ratio, max_ensemble_size)

cleanup(tile)
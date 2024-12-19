from src.setup import setup
from src.xr_to_np import xr_to_np
from src.generate_dataloaders import generate_dataloaders
from src.train_test import train_test
from src.ensemble import ensemble

# Variables
start_month = (1979, 12)
end_month = (1980, 2)
train_test_ratio = 0.2
start_epoch, end_epoch = 0, 5
zip_setting = False
max_ensemble_size = 8

tiles = list(range(0,30))

setup()

xr_to_np(tiles, start_month, end_month, train_test_ratio, zip_setting)

train_dataloader, test_dataloader = generate_dataloaders(tiles, start_month, end_month, train_test_ratio)

train_test(train_dataloader, test_dataloader, start_epoch, end_epoch, focus_tile=24)

ensemble(tiles, start_month, end_month, train_test_ratio, max_ensemble_size)
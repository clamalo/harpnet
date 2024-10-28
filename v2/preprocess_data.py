from src.xr_to_np import xr_to_np

# Variables
start_month = (1979, 10)
end_month = (2022, 9)
train_test_ratio = 0.2
start_epoch, end_epoch = 20, 25
max_ensemble_size = 8

tiles = [7]


for tile in tiles:
    
    xr_to_np(tile, start_month, end_month)
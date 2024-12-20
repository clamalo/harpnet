import random
import numpy as np
import torch
from src.setup import setup
from src.xr_to_np import xr_to_np
from src.generate_dataloaders import generate_dataloaders
from src.train_test import train_test
from src.ensemble import ensemble
from src.constants import TORCH_DEVICE, RANDOM_SEED

# Set seeds once at the start
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Variables
start_month = (1979, 10)
end_month = (1980, 9)
train_test_ratio = 0.2
start_epoch, end_epoch = 0, 5
max_ensemble_size = 8
tiles = [24]

# Adjust this based on your workflow:
zip_setting = 'load'  # Example usage; adjust as needed.

if __name__ == "__main__":
    setup()
    xr_to_np(tiles, start_month, end_month, train_test_ratio, zip_setting=zip_setting)

    if zip_setting == 'save':
        # Data has been zipped and local npy files removed. No data is available locally to train.
        print("Data preprocessed, zipped, and removed. No training performed.")
        exit(0)

    train_dataloader, test_dataloader = generate_dataloaders(tiles, start_month, end_month, train_test_ratio)
    train_test(train_dataloader, test_dataloader, start_epoch, end_epoch, focus_tile=24)
    ensemble(tiles, start_month, end_month, train_test_ratio, max_ensemble_size)
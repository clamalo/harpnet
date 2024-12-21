import random
import numpy as np
import torch
from src.setup import setup
from src.xr_to_np import xr_to_np
from src.generate_dataloaders import generate_dataloaders
from src.train_test import train_test
from src.ensemble import ensemble
from src.constants import TORCH_DEVICE, RANDOM_SEED

# Set seeds once at the start for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Ensure deterministic behavior for reproducible results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Variables controlling the data time range, train/test ratio, and training epochs
# Note: These might no longer be needed if the data is already processed.
start_month = (1979, 10)   # Start year-month for data processing (not used by generate_dataloaders now)
end_month = (1980, 9)      # End year-month for data processing (not used)
train_test_ratio = 0.2     # Fraction of data used for testing (already applied during preprocessing)
start_epoch, end_epoch = 0, 5  # Training epochs range
max_ensemble_size = 8      # Maximum number of models to include in ensemble
# tiles = list(range(0,20))  # Tiles to process (not needed by generate_dataloaders anymore if data is prepped)
tiles = [8, 10]
focus_tile = 8

zip_setting = 'load'  # Options: 'save', 'load', or False

if __name__ == "__main__":
    # Setup environment
    setup()

    # Convert data if needed. If zip_setting='load' and data is already processed, this just loads it.
    xr_to_np(tiles, start_month, end_month, train_test_ratio, zip_setting=zip_setting)

    # If we chose to 'save', data is zipped and removed, no training possible
    if zip_setting == 'save':
        print("Data preprocessed, zipped, and removed. No training performed.")
        exit(0)

    # Generate DataLoaders (no arguments needed now)
    train_dataloader, test_dataloader = generate_dataloaders()

    # Train and test the model
    train_test(train_dataloader, test_dataloader, start_epoch, end_epoch, focus_tile=focus_tile)

    # Create an ensemble
    ensemble(tiles, start_month, end_month, train_test_ratio, max_ensemble_size)
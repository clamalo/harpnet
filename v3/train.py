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
start_month = (1979, 10)   # Start year-month for data processing
end_month = (2022, 9)      # End year-month for data processing
train_test_ratio = 0.2     # Fraction of data used for testing
start_epoch, end_epoch = 0, 10   # Training epochs range
max_ensemble_size = 8      # Maximum number of models to include in ensemble
tiles = list(range(0,20))  # Tiles to process

# Specify zipping behavior for processed data
zip_setting = 'load'  # Options: 'save', 'load', or False

if __name__ == "__main__":
    # Setup the environment (e.g., ensure directories exist)
    setup()

    # Convert raw NetCDF data to Numpy arrays, optionally zip data after processing
    xr_to_np(tiles, start_month, end_month, train_test_ratio, zip_setting=zip_setting)

    # If we chose to 'save', the data is now zipped and local files removed, so no training is possible
    if zip_setting == 'save':
        print("Data preprocessed, zipped, and removed. No training performed.")
        exit(0)

    # Generate DataLoaders for training and testing
    train_dataloader, test_dataloader = generate_dataloaders(tiles, start_month, end_month, train_test_ratio)

    # Train and test the model over the specified epochs
    # Optionally focus on a specific tile (e.g., tile #24) for additional logging
    train_test(train_dataloader, test_dataloader, start_epoch, end_epoch, focus_tile=15)

    # Create an ensemble of the best-performing models
    ensemble(tiles, start_month, end_month, train_test_ratio, max_ensemble_size)
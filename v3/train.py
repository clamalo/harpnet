"""
Main training script.

This script sets up directories, processes or loads data, and then trains the model.
The zip_setting controls how data is handled:

- zip_setting='save': Perform preprocessing, save npy files, then zip them and remove raw npy files. No training is done afterward since files are gone.
- zip_setting='load': Extract data from the zip, and then load it to train.
- zip_setting=False: Just process and store npy files locally without zipping, and then train directly.
"""

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
max_ensemble_size = 8
tiles = [0,6,12,18,24]

# Adjust this based on your workflow:
# 'save' to preprocess locally and zip (no training afterward),
# 'load' to load from zip on remote machine (unzips and trains),
# False to just process and store locally in npy form (then train).
zip_setting = 'load'  # Example usage; adjust as needed.

if __name__ == "__main__":
    setup()
    xr_to_np(tiles, start_month, end_month, train_test_ratio, zip_setting=zip_setting)

    if zip_setting == 'save':
        # Data has been zipped and local npy files removed. No data is available locally to train.
        # The idea is that you'll now take the zip file to the cloud and run with 'load'.
        print("Data preprocessed, zipped, and removed. No training performed.")
        exit(0)

    if zip_setting == 'load':
        # Data was extracted from zip to PROCESSED_DIR, so now we have npy files locally.
        pass  # proceed to load data

    # If zip_setting=False, we have npy files locally after processing.

    train_dataloader, test_dataloader = generate_dataloaders(tiles, start_month, end_month, train_test_ratio)
    train_test(train_dataloader, test_dataloader, start_epoch, end_epoch, focus_tile=24)
    ensemble(tiles, start_month, end_month, train_test_ratio, max_ensemble_size)
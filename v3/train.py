import random
import numpy as np
import torch

from src.setup import setup
from src.xr_to_np import xr_to_np
from src.generate_dataloaders import generate_dataloaders
from src.train_test import train_test
from src.ensemble import ensemble

# Import constants so we don't replicate training controls here
from src.constants import (
    TORCH_DEVICE,
    RANDOM_SEED,
    DATA_START_MONTH,
    DATA_END_MONTH,
    TRAIN_TEST_RATIO,
    TRAIN_START_EPOCH,
    TRAIN_END_EPOCH,
    MAX_ENSEMBLE_SIZE,
    TILES,
    FOCUS_TILE,
    ZIP_SETTING,
)

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


if __name__ == "__main__":
    # Setup environment
    setup()

    # Convert/load data if needed
    xr_to_np(
        TILES,
        DATA_START_MONTH,
        DATA_END_MONTH,
        TRAIN_TEST_RATIO,
        zip_setting=ZIP_SETTING
    )

    # If we chose to 'save', data is zipped and removed, so we skip training
    if ZIP_SETTING == 'save':
        print("Data preprocessed, zipped, and removed. No training performed.")
        exit(0)

    # Generate DataLoaders
    train_dataloader, test_dataloader = generate_dataloaders()

    # Train and test the model
    train_test(
        train_dataloader,
        test_dataloader,
        start_epoch=TRAIN_START_EPOCH,
        end_epoch=TRAIN_END_EPOCH,
        focus_tile=FOCUS_TILE
    )

    # Create an ensemble
    ensemble(
        TILES,
        DATA_START_MONTH,
        DATA_END_MONTH,
        TRAIN_TEST_RATIO,
        max_ensemble_size=MAX_ENSEMBLE_SIZE
    )
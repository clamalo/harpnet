"""
Main training script for the HARPNET project. 
Sets up the environment, performs data preprocessing, and trains the model. 
Also runs an ensemble of saved checkpoints at the end.
"""

import random
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from src.setup import setup
from src.xr_to_np import xr_to_np
from src.generate_dataloaders import generate_dataloaders
from src.train_test import train_test
from src.ensemble import ensemble_checkpoints

# Import constants for controlling the training process
from src.constants import (
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
    DETERMINISTIC,
    CHECKPOINTS_DIR
)

# Set seeds once at the start for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Conditionally set deterministic behavior
if DETERMINISTIC:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

if __name__ == "__main__":
    # --- CREATE REQUIRED DIRECTORIES ---
    setup()

    # --- CONVERT/LOAD DATA IF NECESSARY ---
    xr_to_np()

    # If ZIP_SETTING was 'save', data has been compressed/removed, so skip training
    if ZIP_SETTING == 'save':
        logging.info("Data preprocessed, compressed into NPZ, and removed. No training performed.")
        exit(0)

    # --- BUILD DATALOADERS ---
    train_dataloader, test_dataloader = generate_dataloaders()

    # --- TRAIN AND EVALUATE ---
    train_test(
        train_dataloader,
        test_dataloader,
        start_epoch=TRAIN_START_EPOCH,
        end_epoch=TRAIN_END_EPOCH,
        focus_tile=FOCUS_TILE
    )

    # --- RUN ENSEMBLE ---
    best_ensemble_path = ensemble_checkpoints(
        test_dataloader=test_dataloader,
        device=torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu",
        directory_path=str(CHECKPOINTS_DIR),
        output_path=str(CHECKPOINTS_DIR / "best" / "best_model.pt"),
        max_ensemble_size=MAX_ENSEMBLE_SIZE
    )
    logging.info(f"Best ensemble model saved to: {best_ensemble_path}")
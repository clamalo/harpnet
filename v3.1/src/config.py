"""
Configuration and constants for the HARPNET project.

Use the Config class for all environment-specific constants and hyperparameters.
"""

import torch
from pathlib import Path

class Config:
    # Directory paths
    RAW_DIR = Path("/Volumes/T9/monthly")
    PROCESSED_DIR = Path("/Users/clamalo/documents/harpnet/v3/tiles")
    ZIP_DIR = Path("/Users/clamalo/documents/harpnet/v3/zips")
    CHECKPOINTS_DIR = Path("/Users/clamalo/documents/harpnet/v3/checkpoints")
    FIGURES_DIR = Path("figures")

    # Device configuration
    @staticmethod
    def get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    TORCH_DEVICE = get_device()

    # Model hyperparameters
    UNET_DEPTH = 5
    MODEL_INPUT_CHANNELS = 2
    MODEL_OUTPUT_CHANNELS = 1
    MODEL_OUTPUT_SHAPE = (64, 64)

    # Data grid controls
    HOUR_INCREMENT = 1  # Use 1 for hourly data, 3 for every 3-hour data
    SCALE_FACTOR = 4
    MIN_LAT, MIN_LON = 34, -125
    MAX_LAT, MAX_LON = 51, -104

    # Seed for reproducibility
    RANDOM_SEED = 42
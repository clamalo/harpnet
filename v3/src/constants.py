"""
Constants and configuration values for the HARPNET project.
Paths and hyperparameters are defined here.
"""

import torch
from pathlib import Path

# Directory paths
RAW_DIR = Path("/Volumes/T9/monthly")
PROCESSED_DIR = Path("/Users/clamalo/documents/harpnet/v3/tiles")
ZIP_DIR = Path("/Users/clamalo/documents/harpnet/v3/zips")
CHECKPOINTS_DIR = Path("/Users/clamalo/documents/harpnet/v3/checkpoints")
FIGURES_DIR = Path("figures")

# Device configuration
TORCH_DEVICE = ("cuda" if torch.cuda.is_available() 
                else "mps" if torch.backends.mps.is_available() 
                else "cpu")

# Model hyperparameters
UNET_DEPTH = 5
MODEL_INPUT_CHANNELS = 2
MODEL_OUTPUT_CHANNELS = 1
MODEL_OUTPUT_SHAPE = (64, 64)

# Data grid controls
HOUR_INCREMENT = 1  # 1 for full hourly data, 3 for every 3-hour data
SCALE_FACTOR = 4    # E.g., 8 for 3km, 4 for 6km resolution
MIN_LAT, MIN_LON = 34, -125
MAX_LAT, MAX_LON = 51, -104

# Seed for reproducibility
RANDOM_SEED = 42
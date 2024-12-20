"""
Constants and configuration values for the HARPNET project.
Paths, model hyperparameters, and device settings are defined here.
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
MODEL_NAME = "unetwithattention"
UNET_DEPTH = 2
MODEL_INPUT_CHANNELS = 2
MODEL_OUTPUT_CHANNELS = 1
MODEL_OUTPUT_SHAPE = (64, 64)

# Seed for reproducibility
RANDOM_SEED = 42

# Training run settings
MIN_LAT, MAX_LAT = 34.0, 50.0
MIN_LON, MAX_LON = -125.0, -104.0
HOUR_INCREMENT = 1

TILE_SIZE = 64
COARSE_RESOLUTION = 0.25
FINE_RESOLUTION = 0.0625
PADDING = 0.25
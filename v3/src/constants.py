"""
Constants and configuration values for the HARPNET project.
Paths, model hyperparameters, device settings, and other parameters are defined here.
"""

import torch
from pathlib import Path

# ----------------------------------------------------------
# Directory paths
# ----------------------------------------------------------
RAW_DIR = Path("/Volumes/T9/monthly")
PROCESSED_DIR = Path("/Users/clamalo/documents/harpnet/v3/tiles")
# Normalization stats file (created after xr_to_np finishes)
NORMALIZATION_STATS_FILE = PROCESSED_DIR / "normalization_stats.npy"
CHECKPOINTS_DIR = Path("/Users/clamalo/documents/harpnet/v3/checkpoints")
FIGURES_DIR = Path("figures")

# ----------------------------------------------------------
# Device configuration
# ----------------------------------------------------------
TORCH_DEVICE = (
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)

# ----------------------------------------------------------
# Model hyperparameters
# ----------------------------------------------------------
MODEL_NAME = "unetwithattention"
UNET_DEPTH = 1
MODEL_INPUT_CHANNELS = 2
MODEL_OUTPUT_CHANNELS = 1

# ----------------------------------------------------------
# Seed for reproducibility
# ----------------------------------------------------------
RANDOM_SEED = 42

# ----------------------------------------------------------
# Data and training run settings
# ----------------------------------------------------------
# Geographic domain
MIN_LAT, MAX_LAT = 34.0, 50.0
MIN_LON, MAX_LON = -125.0, -104.0

# Grid resolutions and tile size
HOUR_INCREMENT = 1
TILE_SIZE = 32
COARSE_RESOLUTION = 0.25
FINE_RESOLUTION = 0.125
PADDING = 0.25

# Controls for data/time range and training
DATA_START_MONTH = (1979, 10)  # (year, month)
DATA_END_MONTH = (1989, 9)     # (year, month)
TRAIN_TEST_RATIO = 0.2
TRAIN_START_EPOCH = 0
TRAIN_END_EPOCH = 5
MAX_ENSEMBLE_SIZE = 8
TILES = list(range(0, 20))
FOCUS_TILE = 8

# Whether to save or load data as a compressed NPZ or do neither
# Possible values: 'save', 'load', or False
ZIP_SETTING = 'load'

# ----------------------------------------------------------
# Pre-model interpolation setting
# ----------------------------------------------------------
PRE_MODEL_INTERPOLATION = "nearest"

# ----------------------------------------------------------
# NEW: save_precision
# ----------------------------------------------------------
# Can be either "float32" or "float16". Controls how data are stored on disk.
SAVE_PRECISION = "float16"

# ----------------------------------------------------------
# NEW: Deterministic
# ----------------------------------------------------------
# If True, PyTorch will use fully deterministic operations (which can be slower).
# If False, it will allow some non-deterministic ops but run faster.
DETERMINISTIC = True

# ----------------------------------------------------------
# NEW: Hybrid loss ratio
# ----------------------------------------------------------
# Fraction of the hybrid loss that should come from MSE.
# The rest (1 - MSE_HYBRID_LOSS) will come from MAE.
MSE_HYBRID_LOSS = 1
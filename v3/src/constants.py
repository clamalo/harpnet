"""
Provides globally accessible constants and configurations for the HARPNET project.
This includes paths for data, model hyperparameters, device settings, and other parameters.
All parameters are defined once here to ensure consistency throughout the codebase.
"""

import torch
from pathlib import Path

# ----------------------------------------------------------
# Directory paths
# ----------------------------------------------------------
RAW_DIR = Path("/Volumes/T9/monthly")  # Directory of raw NetCDF data
PROCESSED_DIR = Path("/Users/clamalo/documents/harpnet/v3/tiles")  # Directory for processed Numpy data
NORMALIZATION_STATS_FILE = PROCESSED_DIR / "normalization_stats.npy"  # File storing mean/std for normalization
CHECKPOINTS_DIR = Path("/Users/clamalo/documents/harpnet/v3/checkpoints")  # Directory for saving model checkpoints
FIGURES_DIR = Path("figures")  # Directory for saving plots/figures

# ----------------------------------------------------------
# Device configuration
# ----------------------------------------------------------
TORCH_DEVICE = (
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)  # Automatically select device: CUDA, MPS, or CPU

# ----------------------------------------------------------
# Model hyperparameters
# ----------------------------------------------------------
MODEL_NAME = "unetwithattention"  # Which model file to load (dynamic import)
UNET_DEPTH = 1  # Depth of U-Net architecture
MODEL_INPUT_CHANNELS = 2  # Number of input channels to the model
MODEL_OUTPUT_CHANNELS = 1  # Number of output channels from the model

# ----------------------------------------------------------
# Seed for reproducibility
# ----------------------------------------------------------
RANDOM_SEED = 42  # Fixed seed for ensuring reproducible runs

# ----------------------------------------------------------
# Data and training run settings
# ----------------------------------------------------------
MIN_LAT, MAX_LAT = 34.0, 50.0  # Latitude boundaries for the domain
MIN_LON, MAX_LON = -125.0, -104.0  # Longitude boundaries for the domain
HOUR_INCREMENT = 1  # Time increment (in hours) when extracting data
TILE_SIZE = 32  # Width/height in pixels for each tile
COARSE_RESOLUTION = 0.25  # Coarse spatial resolution in degrees
FINE_RESOLUTION = 0.125  # Fine spatial resolution in degrees
PADDING = 0.25  # Extra padding (in degrees) around each tile for coarse data

DATA_START_MONTH = (1979, 10)  # Start year/month for dataset
DATA_END_MONTH = (1989, 9)     # End year/month for dataset
TRAIN_TEST_RATIO = 0.2         # Fraction of data used for testing
TRAIN_START_EPOCH = 0          # Epoch at which training starts (useful for resume)
TRAIN_END_EPOCH = 5            # Epoch at which training ends
MAX_ENSEMBLE_SIZE = 8          # Maximum number of checkpoints in ensemble
TILES = list(range(0, 20))     # List of tile indices to process
FOCUS_TILE = 8                 # Tile index to focus metrics on (optional)

ZIP_SETTING = 'load'  # Controls whether to load or save compressed data: 'save'|'load'|False
PRE_MODEL_INTERPOLATION = "nearest"  # Interpolation mode for up/downsampling

# ----------------------------------------------------------
# NEW: save_precision
# ----------------------------------------------------------
SAVE_PRECISION = "float16"  # Controls how data are stored on disk: 'float16' or 'float32'

# ----------------------------------------------------------
# NEW: Deterministic
# ----------------------------------------------------------
DETERMINISTIC = True  # If True, enforce fully deterministic operations at a possible performance cost

# ----------------------------------------------------------
# NEW: Hybrid loss ratio
# ----------------------------------------------------------
MSE_HYBRID_LOSS = 1  # Fraction of hybrid loss contributed by MSE; (1 - fraction) is contributed by MAE
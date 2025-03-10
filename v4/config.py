from pathlib import Path
import torch

# -------------------------------------------------------------------
# PATH CONFIGURATIONS
# -------------------------------------------------------------------
RAW_DIR = Path('/Volumes/T9/monthly/')
PROCESSED_DIR = Path('/Users/clamalo/documents/harpnet/v3.1/tiles/')
CHECKPOINTS_DIR = Path('/Users/clamalo/documents/harpnet/v3.1/checkpoints/')
FIGURES_DIR = Path('/Users/clamalo/documents/harpnet/v3.1/figures/')
ELEVATION_FILE = Path('/Users/clamalo/downloads/elevation.nc')

# -------------------------------------------------------------------
# TILE DOMAIN COORDINATES
# -------------------------------------------------------------------
# PRIMARY_MIN_LAT, PRIMARY_MAX_LAT = 33.0, 51.0
# PRIMARY_MIN_LON, PRIMARY_MAX_LON = -126.0, -103.0

# SECONDARY_MIN_LAT, SECONDARY_MAX_LAT = 34.0, 50.0
# SECONDARY_MIN_LON, SECONDARY_MAX_LON = -125.0, -104.0

# PRIMARY_MIN_LAT, PRIMARY_MAX_LAT = 42.0, 52.0
# PRIMARY_MIN_LON, PRIMARY_MAX_LON = 0.0, 16.0

# SECONDARY_MIN_LAT, SECONDARY_MAX_LAT = 43.0, 51.0
# SECONDARY_MIN_LON, SECONDARY_MAX_LON = 1.0, 15.0

# PRIMARY_MIN_LAT, PRIMARY_MAX_LAT = 58.0, 64.0
# PRIMARY_MIN_LON, PRIMARY_MAX_LON = -151.0, -143.0

# SECONDARY_MIN_LAT, SECONDARY_MAX_LAT = 59.0, 63.0
# SECONDARY_MIN_LON, SECONDARY_MAX_LON = -150.0, -144

# PRIMARY_MIN_LAT, PRIMARY_MAX_LAT = 34.0, 47.0
# PRIMARY_MIN_LON, PRIMARY_MAX_LON = 134.0, 147.0

# SECONDARY_MIN_LAT, SECONDARY_MAX_LAT = 35.0, 46.0
# SECONDARY_MIN_LON, SECONDARY_MAX_LON = 135.0, 146.0

PRIMARY_MIN_LAT, PRIMARY_MAX_LAT = 37.0, 43.0
PRIMARY_MIN_LON, PRIMARY_MAX_LON = -115.0, -108.0

SECONDARY_MIN_LAT, SECONDARY_MAX_LAT = 38.0, 42.0
SECONDARY_MIN_LON, SECONDARY_MAX_LON = -114.0, -109.0

# -------------------------------------------------------------------
# TILE SETTINGS
# -------------------------------------------------------------------
TILE_SIZE = 64
COARSE_RESOLUTION = 0.25
FINE_RESOLUTION = 0.03125
PADDING = 0.25

# -------------------------------------------------------------------
# RUNTIME & DEVICE SETTINGS
# -------------------------------------------------------------------
DEVICE = ('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
TRAINING_PROGRESS_BAR = False
ZIP_MODE = "load"
PRECIP_THRESHOLD = 0.01

# -------------------------------------------------------------------
# TRAINING DATA SETTINGS
# -------------------------------------------------------------------
START_MONTH = (1979, 10)
END_MONTH = (1979, 12)
TRAIN_SPLIT = (1979, 12)   # Could also be a float in [0, 1] if desired
TRAINING_TILES = list(range(0,20))
BATCH_SIZE = 64
NUM_EPOCHS = 3
SECONDARY_TILES = True
INCLUDE_ZEROS = False

# -------------------------------------------------------------------
# MODEL HYPERPARAMETERS
# -------------------------------------------------------------------
MODEL_NAME = "unetwithattention"
UNET_DEPTH = 2
MODEL_INPUT_CHANNELS = 2
MODEL_OUTPUT_CHANNELS = 1

# -------------------------------------------------------------------
# ENSEMBLE HYPERPARAMETERS
# -------------------------------------------------------------------
MAX_ENSEMBLE_SIZE = 4

# -------------------------------------------------------------------
# FINE-TUNING PARAMETERS
# -------------------------------------------------------------------
FINE_TUNE_EPOCHS = 5
FINE_TUNE_TILES = [8, 15]
FINE_TUNE_INITIAL_WEIGHTS = CHECKPOINTS_DIR / 'best' / 'best_model.pt'
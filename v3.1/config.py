from pathlib import Path

# Paths
RAW_DIR = Path('/Volumes/T9/monthly/')
PROCESSED_DIR = Path('/Users/clamalo/documents/harpnet/v3.1/tiles/')
CHECKPOINTS_DIR = Path('/Users/clamalo/documents/harpnet/v3.1/checkpoints/')
FIGURES_DIR = Path('/Users/clamalo/documents/harpnet/v3.1/figures/')
ELEVATION_FILE = Path('/Users/clamalo/downloads/elevation.nc')

# Tile domain coordinates
PRIMARY_MIN_LAT, PRIMARY_MAX_LAT = 34.0, 50.0
PRIMARY_MIN_LON, PRIMARY_MAX_LON = -125.0, -104.

SECONDARY_MIN_LAT, SECONDARY_MAX_LAT = 36.0, 48.0
SECONDARY_MIN_LON, SECONDARY_MAX_LON = -123.0, -106.0

# Tile settings
TILE_SIZE = 32
COARSE_RESOLUTION = 0.25
FINE_RESOLUTION = 0.125
PADDING = 0.25

DEVICE = 'mps'

# TRAINING RUN SETTINGS
SECONDARY_TILES = True
SAVE_PRECISION = 'float16'
INCLUDE_ZEROS = True
TRAINING_PROGRESS_BAR = True

BATCH_SIZE = 64
NUM_EPOCHS = 10

# TRAINING DATA SETTINGS
START_MONTH = (1979, 10)
END_MONTH = (1979, 12)
TRAIN_SPLIT = 0.8
# TRAIN_SPLIT = (1980, 2)
TRAINING_TILES = [16]

# Model hyperparameters
MODEL_NAME = "unetwithattention"
UNET_DEPTH = 1

# Updated to 2 channels (e.g., 1 for coarse precipitation, 1 for elevation)
MODEL_INPUT_CHANNELS = 2
MODEL_OUTPUT_CHANNELS = 1
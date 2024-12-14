import torch

RAW_DIR = f'/Users/clamalo/documents/harpnet/v2/monthly'
PROCESSED_DIR = f'/Users/clamalo/documents/harpnet/v2/tiles'
ZIP_DIR = f'/Users/clamalo/documents/harpnet/v2/zips'
CHECKPOINTS_DIR = f'/Users/clamalo/documents/harpnet/v2/v2_checkpoints'
FIGURES_DIR = f'figures'

TORCH_DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# MODEL CONTROLS
UNET_DEPTH = 5

# GRID CONTROLS
HOUR_INCREMENT = 1 # 1 for full, 3 for base
SCALE_FACTOR = 4  # 8 for 3km, 4 for 6km
MIN_LAT, MIN_LON = 30, -125
MAX_LAT, MAX_LON = 51, -104
import random
import numpy as np
import torch

from config import *
from setup import setup
from data_preprocessing import preprocess_data
from generate_dataloaders import generate_dataloaders
from train_test import train_test
from ensemble import ensemble
from fine_tuning import fine_tune_tiles


def set_seed(seed: int = 42):
    """
    Set all relevant random seeds for Python, NumPy, and PyTorch to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set a fixed seed for full reproducibility
set_seed(42)

setup()

if not LOAD:
    preprocess_data()

quit()

train_loader, test_loader = generate_dataloaders()

train_test(train_loader, test_loader)

ensemble(CHECKPOINTS_DIR)

# fine_tune_tiles(TRAINING_TILES, CHECKPOINTS_DIR / 'best' / 'best_model.pt', FINE_TUNE_EPOCHS)
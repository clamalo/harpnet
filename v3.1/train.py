import random
import numpy as np
import torch

from src.utils.setup import setup
from src.data.preprocessing import xr_to_np
from src.data.dataloaders import generate_dataloaders
from src.training.trainer import train_test
from src.training.ensemble import ensemble
from src.config import Config
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

if __name__ == "__main__":
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    start_month = (1979, 10)
    end_month = (1980, 9)
    train_test_ratio = 0.2
    start_epoch, end_epoch = 0, 5
    max_ensemble_size = 8
    tiles = [0,6,12,18,24]

    zip_setting = 'save'  # Example: adjust as needed.

    setup()
    xr_to_np(tiles, start_month, end_month, train_test_ratio, zip_setting=zip_setting)

    if zip_setting == 'save':
        # Data has been zipped and local npy files removed.
        logger.info("Data preprocessed, zipped, and removed. No training performed.")
    else:
        train_dataloader, test_dataloader = generate_dataloaders(tiles, start_month, end_month, train_test_ratio)
        train_test(train_dataloader, test_dataloader, start_epoch, end_epoch, focus_tile=24)
        ensemble(tiles, start_month, end_month, train_test_ratio, max_ensemble_size)
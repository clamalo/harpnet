"""
Script to analyze and print out loss reductions for ensemble models vs bilinear baseline.
"""

import torch
import math
from pathlib import Path
from src.config import Config
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

tiles = range(0, 36)
best_dir = Path("/Users/clamalo/documents/harpnet/v2/best")

rmse_reductions = []
mse_reductions = []
test_losses = []

for tile in tiles:
    ckpt_path = best_dir / f"{tile}_model.pt"
    if not ckpt_path.exists():
        logger.warning(f"No checkpoint found for tile {tile}. Skipping.")
        continue

    checkpoint = torch.load(ckpt_path, map_location=torch.device(Config.TORCH_DEVICE))
    test_loss = checkpoint.get('test_loss', None)
    bilinear_test_loss = checkpoint.get('bilinear_test_loss', None)

    if test_loss is None or bilinear_test_loss is None:
        logger.warning(f"Missing test_loss or bilinear_test_loss in tile {tile} checkpoint.")
        continue

    rmse_reduction = 1 - (math.sqrt(test_loss) / math.sqrt(bilinear_test_loss))
    mse_reduction = 1 - (test_loss / bilinear_test_loss)
    rmse_reductions.append(rmse_reduction)
    mse_reductions.append(mse_reduction)
    test_losses.append(test_loss)

    logger.info(f'Tile {tile} reduction: {rmse_reduction*100:.2f}% RMSE, {mse_reduction*100:.2f}% MSE')
    logger.info(f'Test loss: {test_loss:.4f}, Bilinear test loss: {bilinear_test_loss:.4f}')

if rmse_reductions:
    mean_rmse_reduction = sum(rmse_reductions) / len(rmse_reductions)
    mean_mse_reduction = sum(mse_reductions) / len(mse_reductions)
    mean_test_loss = sum(test_losses) / len(test_losses)

    logger.info(f'Mean RMSE reduction: {mean_rmse_reduction*100:.2f}%')
    logger.info(f'Mean MSE reduction: {mean_mse_reduction*100:.2f}%')
    logger.info(f'Mean test loss: {mean_test_loss:.4f}')
else:
    logger.info("No valid checkpoints processed, no reductions calculated.")
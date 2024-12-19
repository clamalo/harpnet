"""
Utility to sort and select best model checkpoints based on test_loss.
"""

import os
import shutil
import torch
from typing import Optional, List, Union
from src.config import Config
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def sort_epochs(tiles: Optional[List[Union[int,str]]] = None) -> None:
    """
    Sort PyTorch checkpoint files based on test_loss and save the best checkpoint for each tile.

    Args:
        tiles (list): A list of tile numbers/strings. If None, attempts to find all tiles as subdirectories.
    """
    base_dir = Config.CHECKPOINTS_DIR
    best_dir = os.path.join(base_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    if tiles is None:
        try:
            tiles = [
                name for name in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, name)) and name != 'best'
            ]
        except FileNotFoundError:
            logger.error(f"Base directory {base_dir} does not exist.")
            return

    for tile in tiles:
        if tile == 'best':
            logger.info(f"Skipping tile {tile} as it is reserved for best checkpoints.")
            continue

        tile_dir = os.path.join(base_dir, str(tile))
        if not os.path.isdir(tile_dir):
            logger.warning(f"Tile directory {tile_dir} does not exist. Skipping tile {tile}.")
            continue

        checkpoint_files = [f for f in os.listdir(tile_dir) if f.endswith('.pt')]
        if not checkpoint_files:
            logger.info(f"No checkpoint files found in {tile_dir}. Skipping tile {tile}.")
            continue

        checkpoints = []
        for ckpt_file in checkpoint_files:
            ckpt_path = os.path.join(tile_dir, ckpt_file)
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                test_loss = checkpoint.get('test_loss')
                if test_loss is None:
                    logger.warning(f"'test_loss' not found in {ckpt_path}. Skipping.")
                    continue
                checkpoints.append((ckpt_file, test_loss))
            except Exception as e:
                logger.error(f"Error loading {ckpt_path}: {e}. Skipping.")
                continue

        if not checkpoints:
            logger.info(f"No valid checkpoints with 'test_loss' found in {tile_dir}.")
            continue

        # Sort by test_loss
        checkpoints.sort(key=lambda x: x[1])

        best_ckpt_file, best_test_loss = checkpoints[0]
        best_ckpt_path = os.path.join(tile_dir, best_ckpt_file)
        destination_path = os.path.join(best_dir, f"{tile}_model.pt")

        try:
            shutil.copyfile(best_ckpt_path, destination_path)
            logger.info(f"Best checkpoint for tile {tile} (test_loss: {best_test_loss}) saved to {destination_path}.")
        except Exception as e:
            logger.error(f"Failed to copy {best_ckpt_path} to {destination_path}: {e}.")
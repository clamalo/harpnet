"""
Utility to clean up processed tile directories.
"""

import shutil
from pathlib import Path
from src.config import Config
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def cleanup(tile: int) -> None:
    """
    Remove the processed directory for a specific tile.

    Args:
        tile (int): Tile index to remove.
    """
    tile_dir = Path(Config.PROCESSED_DIR) / f"{tile}"
    if tile_dir.exists():
        shutil.rmtree(str(tile_dir))
        logger.info(f"Removed processed directory for tile {tile}: {tile_dir}")
    else:
        logger.warning(f"Tile directory does not exist: {tile_dir}")
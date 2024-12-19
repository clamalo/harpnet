"""
Utility to clean up processed tile directories.
"""

import shutil
from src.constants import PROCESSED_DIR
from pathlib import Path

def cleanup(tile: int) -> None:
    """
    Remove the processed directory for a specific tile.
    """
    tile_dir = Path(PROCESSED_DIR) / f"{tile}"
    if tile_dir.exists():
        shutil.rmtree(str(tile_dir))
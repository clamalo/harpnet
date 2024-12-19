import os

from src.constants import PROCESSED_DIR, ZIP_DIR, CHECKPOINTS_DIR, FIGURES_DIR

def setup():
    """
    Setup directories for a multi-tile training scenario.
    No longer creates tile-specific directories, just ensures base dirs exist.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ZIP_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
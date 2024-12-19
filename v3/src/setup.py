"""
Setup script to ensure required directories exist.
"""

import os
from src.constants import PROCESSED_DIR, ZIP_DIR, CHECKPOINTS_DIR, FIGURES_DIR

def setup():
    """
    Setup base directories for processing data.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ZIP_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
"""
Ensures required directories exist for data processing and output.
"""

import os
from src.constants import PROCESSED_DIR, CHECKPOINTS_DIR, FIGURES_DIR

def setup():
    """
    Create base directories if they do not exist.
    Ensures a consistent file structure for the project.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
"""
Initializes the project's directory structure if needed.
Ensures that all required directories exist for file outputs and data storage.
"""

import os
from src.constants import PROCESSED_DIR, CHECKPOINTS_DIR, FIGURES_DIR

def setup():
    """
    Creates the directories used for storing processed data, checkpoints, and figures.
    If they already exist, no action is taken.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
import os

from src.constants import PROCESSED_DIR, ZIP_DIR, CHECKPOINTS_DIR, FIGURES_DIR


def setup():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ZIP_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # figures
    os.makedirs(FIGURES_DIR, exist_ok=True)
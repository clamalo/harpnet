import os

from src.constants import PROCESSED_DIR, ZIP_DIR, CHECKPOINTS_DIR, FIGURES_DIR, ZIP


def setup(tile):
    os.makedirs(os.path.join(PROCESSED_DIR, str(tile)), exist_ok=True)
    os.makedirs(ZIP_DIR, exist_ok=True)
    os.makedirs(os.path.join(CHECKPOINTS_DIR, str(tile)), exist_ok=True)

    # figures
    os.makedirs(FIGURES_DIR, exist_ok=True)
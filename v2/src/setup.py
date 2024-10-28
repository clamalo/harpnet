import os
import zipfile

from src.constants import PROCESSED_DIR, CHECKPOINTS_DIR, FIGURES_DIR, ZIP


def setup(tile):
    os.makedirs(os.path.join(PROCESSED_DIR, str(tile)), exist_ok=True)
    os.makedirs(os.path.join(CHECKPOINTS_DIR, str(tile)), exist_ok=True)

    # figures
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if ZIP and os.path.exists(os.path.join(PROCESSED_DIR, f"{tile}.zip")):
        with zipfile.ZipFile(os.path.join(PROCESSED_DIR, f"{tile}.zip"), 'r') as zip_ref:
            zip_ref.extractall(PROCESSED_DIR)
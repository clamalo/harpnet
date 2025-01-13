import random
import numpy as np
import torch
import shutil
import os

from config import *
from setup import setup
from data_preprocessing import preprocess_data
from generate_dataloaders import generate_dataloaders
from train_test import train_test
from ensemble import ensemble
from fine_tuning import fine_tune_tiles


def set_seed(seed: int = 42):
    """
    Set all relevant random seeds for Python, NumPy, and PyTorch to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set a fixed seed for full reproducibility
set_seed(42)

setup()

if ZIP_MODE == "save":
    """
    1) Preprocess data (including memmapped final train/test).
    2) Zip up everything in PROCESSED_DIR into data.zip, excluding data.zip itself.
    3) Exit.
    """
    preprocess_data()

    zip_path = PROCESSED_DIR / "data.zip"
    if zip_path.exists():
        # If there's an old data.zip, remove it so we don't re-include it in the new one.
        os.remove(zip_path)

    # Create a temporary directory to stage the files we want to zip.
    temp_dir = PROCESSED_DIR / "tmp_zippable"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    # Copy everything from PROCESSED_DIR into temp_dir, except data.zip itself
    # (and the temp_dir of course, but that's not created yet).
    shutil.copytree(
        src=PROCESSED_DIR,
        dst=temp_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("data.zip", "tmp_zippable")
    )

    print(f"Zipping {PROCESSED_DIR} into {zip_path} ...")
    # Now zip the contents of temp_dir into data.zip
    shutil.make_archive(
        base_name=str(PROCESSED_DIR / "data"),  # 'tiles/data'
        format='zip',
        root_dir=str(temp_dir),
        base_dir='.'
    )
    # Clean up the temp staging dir
    shutil.rmtree(temp_dir)

    print(f"Created {zip_path}")
    quit()

elif ZIP_MODE == "load":
    """
    1) Unzip data.zip within PROCESSED_DIR, restoring final data/metadata.
    2) Train, test, or do ensemble/fine-tuning as desired.
    """
    zip_path = PROCESSED_DIR / "data.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Cannot find {zip_path} to unzip. Make sure you've placed data.zip in {PROCESSED_DIR}.")

    # Unzip data.zip in place
    print(f"Unzipping {zip_path} into {PROCESSED_DIR} ...")
    shutil.unpack_archive(str(zip_path), extract_dir=str(PROCESSED_DIR), format='zip')
    print("Unzipped successfully. Starting training...")

    # Now proceed with training
    train_loader, test_loader = generate_dataloaders()
    train_test(train_loader, test_loader)

    # ensemble(CHECKPOINTS_DIR)

    # fine_tune_tiles(TRAINING_TILES, CHECKPOINTS_DIR / 'best' / 'best_model.pt', FINE_TUNE_EPOCHS)

else:
    raise ValueError(f"ZIP_MODE must be either 'save' or 'load', but got '{ZIP_MODE}'.")
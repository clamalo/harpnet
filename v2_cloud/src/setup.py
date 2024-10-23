import os
import zipfile

import src.constants as constants

def setup(tile):
    if not os.path.exists(os.path.join(constants.processed_dir, str(tile))):
        os.makedirs(os.path.join(constants.processed_dir, str(tile)))
    if not os.path.exists(os.path.join(constants.checkpoints_dir, str(tile))):
        os.makedirs(os.path.join(constants.checkpoints_dir, str(tile)))

    # figures
    if not os.path.exists(constants.figures_dir):
        os.makedirs(constants.figures_dir)

    # unzip tile zip file from processed into processed directory
    # e.g. 4.zip -> 4/
    with zipfile.ZipFile(os.path.join(constants.processed_dir, f"{tile}.zip"), 'r') as zip_ref:
        zip_ref.extractall(constants.processed_dir)
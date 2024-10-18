import os
import src.constants as constants

def setup(tile):
    if not os.path.exists(os.path.join(constants.processed_dir, tile)):
        os.makedirs(os.path.join(constants.processed_dir, tile))
    if not os.path.exists(os.path.join(constants.checkpoints_dir, tile)):
        os.makedirs(os.path.join(constants.checkpoints_dir, tile))

    # figures
    if not os.path.exists(constants.figures_dir):
        os.makedirs(constants.figures_dir)
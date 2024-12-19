import shutil

from src.constants import PROCESSED_DIR

def cleanup(tile):
    shutil.rmtree(f'{PROCESSED_DIR}/{tile}')
import os

def setup(tile):
    if not os.path.exists(f'/Volumes/T9/tiles/{tile}'):
        os.makedirs(f'/Volumes/T9/tiles/{tile}')
    if not os.path.exists(f'/Volumes/T9/v2_checkpoints/{tile}'):
        os.makedirs(f'/Volumes/T9/v2_checkpoints/{tile}')

    # figures
    if not os.path.exists(f'figures'):
        os.makedirs(f'figures')
import os
import shutil
import torch
from tqdm import tqdm
from src.constants import CHECKPOINTS_DIR

def sort_epochs(tiles=None):
    """
    Sort PyTorch checkpoint files based on test_loss and save the best checkpoint for each tile.

    Args:
        tiles (list, optional): A list of tile numbers to process. If None, all tiles in the
                                /Volumes/T9/v2_checkpoints/ directory will be processed.
    """
    base_dir = CHECKPOINTS_DIR
    best_dir = os.path.join(base_dir, "best")

    # Create the 'best' directory if it doesn't exist
    os.makedirs(best_dir, exist_ok=True)

    # If tiles are not provided, list all subdirectories in the base_dir
    if tiles is None:
        try:
            tiles = [name for name in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, name))]
        except FileNotFoundError:
            print(f"Base directory {base_dir} does not exist.")
            return

    for tile in tiles:

        if tile == 'best':
            print(f"Skipping tile {tile} as it is reserved for storing the best checkpoints.")
            continue

        tile_dir = os.path.join(base_dir, str(tile))
        if not os.path.isdir(tile_dir):
            print(f"Tile directory {tile_dir} does not exist. Skipping tile {tile}.")
            continue

        # List all .pt files in the tile directory
        checkpoint_files = [f for f in os.listdir(tile_dir) if f.endswith('.pt')]

        if not checkpoint_files:
            print(f"No checkpoint files found in {tile_dir}. Skipping tile {tile}.")
            continue

        checkpoints = []
        
        for ckpt_file in tqdm(checkpoint_files):
            ckpt_path = os.path.join(tile_dir, ckpt_file)
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                test_loss = checkpoint.get('test_loss')
                
                if test_loss is None:
                    print(f"'test_loss' not found in {ckpt_path}. Skipping this checkpoint.")
                    continue

                checkpoints.append((ckpt_file, test_loss))
            except Exception as e:
                print(f"Error loading {ckpt_path}: {e}. Skipping this checkpoint.")
                continue

        if not checkpoints:
            print(f"No valid checkpoints with 'test_loss' found in {tile_dir}.")
            continue

        # Sort the checkpoints by test_loss in ascending order
        checkpoints.sort(key=lambda x: x[1])

        best_ckpt_file, best_test_loss = checkpoints[0]
        best_ckpt_path = os.path.join(tile_dir, best_ckpt_file)
        destination_path = os.path.join(best_dir, f"{tile}_model.pt")

        try:
            shutil.copyfile(best_ckpt_path, destination_path)
            print(f"Best checkpoint for tile {tile} (test_loss: {best_test_loss}) saved to {destination_path}.")
        except Exception as e:
            print(f"Failed to copy {best_ckpt_path} to {destination_path}: {e}.")


if __name__ == "__main__":
    sort_epochs()
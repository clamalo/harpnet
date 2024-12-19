from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from src.constants import PROCESSED_DIR

def generate_dataloaders(tiles, first_month, last_month, train_test_ratio):
    """
    Generate dataloaders for multiple tiles at once. This function will load
    input, target, times, and tile arrays from all specified tiles, concatenate them,
    shuffle them, and then split into train and test sets.
    """

    class MemMapDataset(Dataset):
        def __init__(self, inputs, targets, times, tiles):
            self.inputs = inputs
            self.targets = targets
            self.times = times
            self.tiles = tiles
        def __len__(self):
            return self.inputs.shape[0]
        def __getitem__(self, idx):
            return self.inputs[idx], self.targets[idx], self.times[idx], self.tiles[idx]

    # Ensure tiles is a list
    if isinstance(tiles, int):
        tiles = [tiles]

    # Generate list of months between first_month and last_month inclusive
    first = datetime(*first_month, day=1)
    last = datetime(*last_month, day=1)
    months = []
    current = first
    while current <= last:
        months.append(current)
        current += relativedelta(months=1)

    # Collect file paths from all tiles
    input_file_paths = []
    target_file_paths = []
    times_file_paths = []
    tile_file_paths = []

    for tile in tiles:
        for m in months:
            input_fp = os.path.join(PROCESSED_DIR, str(tile), f'input_{m.year}_{m.month:02d}.npy')
            target_fp = os.path.join(PROCESSED_DIR, str(tile), f'target_{m.year}_{m.month:02d}.npy')
            times_fp = os.path.join(PROCESSED_DIR, str(tile), f'times_{m.year}_{m.month:02d}.npy')
            tile_fp = os.path.join(PROCESSED_DIR, str(tile), f'tile_{m.year}_{m.month:02d}.npy')

            if os.path.exists(input_fp) and os.path.exists(target_fp) and os.path.exists(times_fp) and os.path.exists(tile_fp):
                input_file_paths.append(input_fp)
                target_file_paths.append(target_fp)
                times_file_paths.append(times_fp)
                tile_file_paths.append(tile_fp)

    if not input_file_paths:
        raise FileNotFoundError("No input files found for the given tiles and month range.")

    # Load and concatenate arrays
    input_arr = np.concatenate([np.load(fp) for fp in input_file_paths])
    target_arr = np.concatenate([np.load(fp) for fp in target_file_paths])
    times_arr = np.concatenate([np.load(fp) for fp in times_file_paths]).astype('datetime64[s]').astype(np.float64)
    tile_arr = np.concatenate([np.load(fp) for fp in tile_file_paths])

    # Shuffle the data
    np.random.seed(42)
    indices = np.random.permutation(len(times_arr))
    input_arr, target_arr, times_arr, tile_arr = input_arr[indices], target_arr[indices], times_arr[indices], tile_arr[indices]

    print(times_arr[:3].astype('datetime64[s]'))

    # Split the data
    split_idx = int(train_test_ratio * len(input_arr))
    train_input, test_input = input_arr[split_idx:], input_arr[:split_idx]
    train_target, test_target = target_arr[split_idx:], target_arr[:split_idx]
    train_times, test_times = times_arr[split_idx:], times_arr[:split_idx]
    train_tiles_arr, test_tiles_arr = tile_arr[split_idx:], tile_arr[:split_idx]

    # Create datasets
    train_dataset = MemMapDataset(train_input, train_target, train_times, train_tiles_arr)
    test_dataset = MemMapDataset(test_input, test_target, test_times, test_tiles_arr)

    # Create DataLoaders with deterministic shuffling for training
    generator = torch.Generator()
    generator.manual_seed(42)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader
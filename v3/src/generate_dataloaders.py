"""
Generates PyTorch dataloaders for training and testing.
Loads previously processed coarse and fine inputs, targets, times, and tile IDs.
Also provides elevation data for each tile.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from src.constants import PROCESSED_DIR, RANDOM_SEED
from typing import List, Tuple

def generate_dataloaders(tiles: List[int], 
                         first_month: Tuple[int,int], 
                         last_month: Tuple[int,int], 
                         train_test_ratio: float):
    """
    Generate training and testing dataloaders based on preprocessed Numpy arrays.

    Args:
        tiles: List of tile indices.
        first_month: (year, month) start period.
        last_month: (year, month) end period.
        train_test_ratio: Ratio for splitting train/test data (handled previously).

    Returns:
        train_dataloader, test_dataloader
    """

    class CombinedDataset(Dataset):
        """
        Combined dataset that contains inputs, targets, times, and tile IDs for multiple tiles.
        Also includes elevation data for each tile.
        """
        def __init__(self, inputs: np.ndarray, 
                     targets: np.ndarray, 
                     times: np.ndarray, 
                     tile_ids: np.ndarray, 
                     tile_elev: np.ndarray, 
                     tile_id_to_index: dict):
            self.inputs = inputs
            self.targets = targets
            self.times = times
            self.tile_ids = tile_ids
            self.tile_elev = tile_elev
            self.tile_id_to_index = tile_id_to_index

        def __len__(self) -> int:
            return self.inputs.shape[0]

        def __getitem__(self, idx: int):
            input_data = torch.from_numpy(self.inputs[idx])   # (C,H,W)
            target_data = torch.from_numpy(self.targets[idx]) # (1,H,W)
            time_data = self.times[idx]
            tile_data = self.tile_ids[idx]

            tile_idx = self.tile_id_to_index[tile_data]
            elev_data = torch.from_numpy(self.tile_elev[tile_idx]) # (1,Hf,Wf)

            return input_data, elev_data, target_data, time_data, tile_data

    # Load preprocessed data arrays
    train_input_path = PROCESSED_DIR / "combined_train_input.npy"
    train_target_path = PROCESSED_DIR / "combined_train_target.npy"
    train_times_path = PROCESSED_DIR / "combined_train_times.npy"
    train_tile_ids_path = PROCESSED_DIR / "combined_train_tile_ids.npy"

    test_input_path = PROCESSED_DIR / "combined_test_input.npy"
    test_target_path = PROCESSED_DIR / "combined_test_target.npy"
    test_times_path = PROCESSED_DIR / "combined_test_times.npy"
    test_tile_ids_path = PROCESSED_DIR / "combined_test_tile_ids.npy"

    tile_elev_path = PROCESSED_DIR / "combined_tile_elev.npy"

    train_input = np.load(train_input_path)
    train_target = np.load(train_target_path)
    train_times = np.load(train_times_path)
    train_tile_ids = np.load(train_tile_ids_path)

    test_input = np.load(test_input_path)
    test_target = np.load(test_target_path)
    test_times = np.load(test_times_path)
    test_tile_ids = np.load(test_tile_ids_path)

    tile_elev = np.load(tile_elev_path)  # (num_tiles,1,Hf,Wf)

    unique_tile_ids = sorted(set(train_tile_ids) | set(test_tile_ids))
    tile_id_to_index = {t: i for i, t in enumerate(unique_tile_ids)}

    train_dataset = CombinedDataset(train_input, train_target, train_times, train_tile_ids, tile_elev, tile_id_to_index)
    test_dataset = CombinedDataset(test_input, test_target, test_times, test_tile_ids, tile_elev, tile_id_to_index)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(RANDOM_SEED)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=loader_generator, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader
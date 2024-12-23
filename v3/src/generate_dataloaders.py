"""
Generates PyTorch dataloaders for training and testing.
Loads previously processed arrays, applies normalization, and can optionally focus on a single tile.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from src.constants import PROCESSED_DIR, RANDOM_SEED, NORMALIZATION_STATS_FILE, TILE_SIZE, PRE_MODEL_INTERPOLATION
from typing import List, Tuple, Optional

class CombinedDataset(Dataset):
    """
    Combined dataset that contains inputs, targets, times, and tile IDs for multiple tiles.
    Also includes elevation data for each tile, and applies normalization after interpolation.

    If focus_tile is provided, the dataset is filtered to include only that tile.
    """
    def __init__(self, 
                 inputs: np.ndarray, 
                 targets: np.ndarray, 
                 times: np.ndarray, 
                 tile_ids: np.ndarray, 
                 tile_elev: np.ndarray, 
                 tile_id_to_index: dict,
                 mean_val: float,
                 std_val: float,
                 focus_tile: Optional[int] = None):
        if focus_tile is not None:
            mask = (tile_ids == focus_tile)
            inputs = inputs[mask]
            targets = targets[mask]
            times = times[mask]
            tile_ids = tile_ids[mask]

        self.inputs = inputs
        self.targets = targets
        self.times = times
        self.tile_ids = tile_ids
        self.tile_elev = tile_elev
        self.tile_id_to_index = tile_id_to_index
        self.mean = mean_val
        self.std = std_val

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        input_data = torch.from_numpy(self.inputs[idx])   # (C,H,W)
        target_data = torch.from_numpy(self.targets[idx]) # (1,H,W)
        time_data = self.times[idx]
        tile_data = self.tile_ids[idx]

        tile_idx = self.tile_id_to_index[tile_data]
        elev_data = torch.from_numpy(self.tile_elev[tile_idx]) # (1,Hf,Wf)

        # Interpolate to TILE_SIZE x TILE_SIZE
        input_data = torch.nn.functional.interpolate(input_data.unsqueeze(0), size=(TILE_SIZE,TILE_SIZE), mode=PRE_MODEL_INTERPOLATION).squeeze(0)
        elev_data = torch.nn.functional.interpolate(elev_data.unsqueeze(0), size=(TILE_SIZE,TILE_SIZE), mode=PRE_MODEL_INTERPOLATION).squeeze(0)
        target_data = torch.nn.functional.interpolate(target_data.unsqueeze(0), size=(TILE_SIZE,TILE_SIZE), mode=PRE_MODEL_INTERPOLATION).squeeze(0)

        # Combine inputs with elevation
        input_data = torch.cat([input_data, elev_data], dim=0) # (2,H,W)

        # Normalize precipitation channels (input_data[0,:,:] is precip)
        input_data[0,:,:] = (input_data[0,:,:] - self.mean) / self.std
        target_data = (target_data - self.mean) / self.std

        return input_data, elev_data, target_data, time_data, tile_data

def load_normalization_stats():
    norm_stats = np.load(NORMALIZATION_STATS_FILE)
    mean_val, std_val = float(norm_stats[0]), float(norm_stats[1])
    return mean_val, std_val

def generate_dataloaders(focus_tile: Optional[int] = None):
    """
    Generate training and testing dataloaders based on preprocessed Numpy arrays.
    If focus_tile is provided, returns loaders filtered to that single tile.
    Otherwise, returns loaders for all tiles.
    """

    mean_val, std_val = load_normalization_stats()

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

    train_dataset = CombinedDataset(train_input, train_target, train_times, train_tile_ids, tile_elev, tile_id_to_index, mean_val, std_val, focus_tile=focus_tile)
    test_dataset = CombinedDataset(test_input, test_target, test_times, test_tile_ids, tile_elev, tile_id_to_index, mean_val, std_val, focus_tile=focus_tile)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(RANDOM_SEED)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=loader_generator, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader
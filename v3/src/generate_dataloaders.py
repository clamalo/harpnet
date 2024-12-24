"""
Generates PyTorch Dataloaders for the project. 
Pulls data from preprocessed Numpy arrays on disk, applies normalization, and optionally filters by tile.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from src.constants import PROCESSED_DIR, RANDOM_SEED, NORMALIZATION_STATS_FILE, TILE_SIZE, PRE_MODEL_INTERPOLATION, BATCH_SIZE
from typing import List, Tuple, Optional

class CombinedDataset(Dataset):
    """
    Custom Dataset that holds inputs, targets, times, and tile IDs for multiple tiles.
    Also handles elevation data and normalizes precipitation channels.
    """
    def __init__(
        self,
        inputs: np.ndarray, 
        targets: np.ndarray, 
        times: np.ndarray, 
        tile_ids: np.ndarray, 
        tile_elev: np.ndarray, 
        tile_id_to_index: dict,
        mean_val: float,
        std_val: float,
        focus_tile: Optional[int] = None
    ):
        # --- FILTER TO FOCUS TILE IF PROVIDED ---
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
        # --- FETCH AND CAST TO TENSORS ---
        input_data = torch.from_numpy(self.inputs[idx])   # (C,H,W)
        target_data = torch.from_numpy(self.targets[idx]) # (1,H,W)
        time_data = self.times[idx]
        tile_data = self.tile_ids[idx]

        tile_idx = self.tile_id_to_index[tile_data]
        elev_data = torch.from_numpy(self.tile_elev[tile_idx]) # (1,Hf,Wf)

        # --- INTERPOLATE TO TILE_SIZE ---
        input_data = torch.nn.functional.interpolate(
            input_data.unsqueeze(0),
            size=(TILE_SIZE, TILE_SIZE),
            mode=PRE_MODEL_INTERPOLATION
        ).squeeze(0)

        elev_data = torch.nn.functional.interpolate(
            elev_data.unsqueeze(0),
            size=(TILE_SIZE, TILE_SIZE),
            mode=PRE_MODEL_INTERPOLATION
        ).squeeze(0)

        target_data = torch.nn.functional.interpolate(
            target_data.unsqueeze(0),
            size=(TILE_SIZE, TILE_SIZE),
            mode=PRE_MODEL_INTERPOLATION
        ).squeeze(0)

        # --- COMBINE INPUT CHANNELS & NORMALIZE ---
        input_data = torch.cat([input_data, elev_data], dim=0)  # (2,H,W)
        input_data[0,:,:] = (input_data[0,:,:] - self.mean) / self.std
        target_data = (target_data - self.mean) / self.std

        return input_data, elev_data, target_data, time_data, tile_data

def load_normalization_stats():
    """
    Loads mean and std from the saved normalization_stats.npy file.
    Raises FileNotFoundError if it does not exist.
    """
    if not NORMALIZATION_STATS_FILE.exists():
        raise FileNotFoundError(
            f"Normalization stats file {NORMALIZATION_STATS_FILE} not found. Did you run xr_to_np?"
        )
    norm_stats = np.load(NORMALIZATION_STATS_FILE)
    mean_val, std_val = float(norm_stats[0]), float(norm_stats[1])
    return mean_val, std_val

def generate_dataloaders(focus_tile: Optional[int] = None):
    """
    Generates training and testing DataLoaders from preprocessed Numpy arrays.
    If focus_tile is set, only data from that tile is included in the dataset.
    """
    # --- CHECK IF DATA FILES EXIST ---
    train_input_path = PROCESSED_DIR / "combined_train_input.npy"
    test_input_path = PROCESSED_DIR / "combined_test_input.npy"
    if not train_input_path.exists() or not test_input_path.exists():
        raise FileNotFoundError(
            f"One of {train_input_path} or {test_input_path} not found. Did you run xr_to_np?"
        )

    # --- LOAD NORMALIZATION STATS ---
    mean_val, std_val = load_normalization_stats()

    # --- LOAD DATA FROM DISK ---
    train_input = np.load(PROCESSED_DIR / "combined_train_input.npy").astype(np.float32)
    train_target = np.load(PROCESSED_DIR / "combined_train_target.npy").astype(np.float32)
    train_times = np.load(PROCESSED_DIR / "combined_train_times.npy")
    train_tile_ids = np.load(PROCESSED_DIR / "combined_train_tile_ids.npy")

    test_input = np.load(PROCESSED_DIR / "combined_test_input.npy").astype(np.float32)
    test_target = np.load(PROCESSED_DIR / "combined_test_target.npy").astype(np.float32)
    test_times = np.load(PROCESSED_DIR / "combined_test_times.npy")
    test_tile_ids = np.load(PROCESSED_DIR / "combined_test_tile_ids.npy")

    tile_elev = np.load(PROCESSED_DIR / "combined_tile_elev.npy").astype(np.float32)

    # --- BUILD TILE ID -> INDEX MAPPING ---
    unique_tile_ids = sorted(set(train_tile_ids) | set(test_tile_ids))
    tile_id_to_index = {t: i for i, t in enumerate(unique_tile_ids)}

    # --- CREATE DATASETS ---
    train_dataset = CombinedDataset(
        train_input, train_target, train_times, train_tile_ids,
        tile_elev, tile_id_to_index, mean_val, std_val,
        focus_tile=focus_tile
    )

    test_dataset = CombinedDataset(
        test_input, test_target, test_times, test_tile_ids,
        tile_elev, tile_id_to_index, mean_val, std_val,
        focus_tile=focus_tile
    )

    # --- CREATE DATALOADERS WITH SEED ---
    loader_generator = torch.Generator()
    loader_generator.manual_seed(RANDOM_SEED)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,  # controlled from constants.py
        shuffle=True,
        generator=loader_generator,
        num_workers=0
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,  # controlled from constants.py
        shuffle=False,
        num_workers=0
    )

    return train_dataloader, test_dataloader
"""
Functions for generating PyTorch dataloaders from preprocessed datasets.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple
from pathlib import Path
from src.config import Config
from src.data.dataset import CombinedDataset

def generate_dataloaders(
    tiles: List[int],
    first_month: Tuple[int,int],
    last_month: Tuple[int,int],
    train_test_ratio: float
) -> Tuple[DataLoader, DataLoader]:
    """
    Generate training and testing dataloaders for given tiles and date ranges.

    Args:
        tiles (List[int]): List of tile indices.
        first_month (Tuple[int,int]): (year, month) for start.
        last_month (Tuple[int,int]): (year, month) for end.
        train_test_ratio (float): Train/test split ratio.

    Returns:
        (DataLoader, DataLoader): train and test dataloaders
    """

    train_input_path = Config.PROCESSED_DIR / "combined_train_input.npy"
    train_target_path = Config.PROCESSED_DIR / "combined_train_target.npy"
    train_times_path = Config.PROCESSED_DIR / "combined_train_times.npy"
    train_tile_ids_path = Config.PROCESSED_DIR / "combined_train_tile_ids.npy"

    test_input_path = Config.PROCESSED_DIR / "combined_test_input.npy"
    test_target_path = Config.PROCESSED_DIR / "combined_test_target.npy"
    test_times_path = Config.PROCESSED_DIR / "combined_test_times.npy"
    test_tile_ids_path = Config.PROCESSED_DIR / "combined_test_tile_ids.npy"

    tile_elev_path = Config.PROCESSED_DIR / "combined_tile_elev.npy"

    # Load arrays
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

    train_dataset = CombinedDataset(
        train_input, train_target, train_times, train_tile_ids, tile_elev, tile_id_to_index
    )
    test_dataset = CombinedDataset(
        test_input, test_target, test_times, test_tile_ids, tile_elev, tile_id_to_index
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(Config.RANDOM_SEED)

    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, generator=loader_generator, num_workers=0
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    return train_dataloader, test_dataloader
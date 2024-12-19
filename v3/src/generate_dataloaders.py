import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from src.constants import PROCESSED_DIR

def generate_dataloaders(tiles, first_month, last_month, train_test_ratio):
    """
    Loads the combined train/test sets produced by xr_to_np, including elevation data.
    Returns train and test dataloaders with (input, elevation, target, time, tile_id).
    """

    class CombinedDataset(Dataset):
        def __init__(self, inputs, targets, times, tile_ids, elev):
            self.inputs = inputs
            self.targets = targets
            self.times = times  # int64 (seconds since epoch)
            self.tile_ids = tile_ids
            self.elev = elev
        def __len__(self):
            return self.inputs.shape[0]
        def __getitem__(self, idx):
            input_data = torch.from_numpy(self.inputs[idx])   # (C,H,W)
            target_data = torch.from_numpy(self.targets[idx]) # (1,H,W)
            elev_data = torch.from_numpy(self.elev[idx])      # (1,Hf,Wf), same resolution as targets
            time_data = self.times[idx]                       # int64
            tile_data = self.tile_ids[idx]
            return input_data, elev_data, target_data, time_data, tile_data

    train_input_path = os.path.join(PROCESSED_DIR, "combined_train_input.npy")
    train_target_path = os.path.join(PROCESSED_DIR, "combined_train_target.npy")
    train_times_path = os.path.join(PROCESSED_DIR, "combined_train_times.npy")
    train_tile_ids_path = os.path.join(PROCESSED_DIR, "combined_train_tile_ids.npy")
    train_elev_path = os.path.join(PROCESSED_DIR, "combined_train_elev.npy")

    test_input_path = os.path.join(PROCESSED_DIR, "combined_test_input.npy")
    test_target_path = os.path.join(PROCESSED_DIR, "combined_test_target.npy")
    test_times_path = os.path.join(PROCESSED_DIR, "combined_test_times.npy")
    test_tile_ids_path = os.path.join(PROCESSED_DIR, "combined_test_tile_ids.npy")
    test_elev_path = os.path.join(PROCESSED_DIR, "combined_test_elev.npy")

    train_input = np.load(train_input_path)
    train_target = np.load(train_target_path)
    train_times = np.load(train_times_path)
    train_tile_ids = np.load(train_tile_ids_path)
    train_elev = np.load(train_elev_path)

    test_input = np.load(test_input_path)
    test_target = np.load(test_target_path)
    test_times = np.load(test_times_path)
    test_tile_ids = np.load(test_tile_ids_path)
    test_elev = np.load(test_elev_path)

    train_dataset = CombinedDataset(train_input, train_target, train_times, train_tile_ids, train_elev)
    test_dataset = CombinedDataset(test_input, test_target, test_times, test_tile_ids, test_elev)

    # Use a fixed seed for deterministic shuffling
    loader_generator = torch.Generator().manual_seed(42)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=loader_generator)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader

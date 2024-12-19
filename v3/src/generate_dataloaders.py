import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from src.constants import PROCESSED_DIR

def generate_dataloaders(tiles, first_month, last_month, train_test_ratio):
    """
    Modified to handle elevation differently:
    We now load a single elevation grid per tile from combined_tile_elev.npy.
    The dataset then uses tile_id to index into this elevation array.
    """

    class CombinedDataset(Dataset):
        def __init__(self, inputs, targets, times, tile_ids, tile_elev, tile_id_to_index):
            self.inputs = inputs
            self.targets = targets
            self.times = times
            self.tile_ids = tile_ids
            self.tile_elev = tile_elev
            self.tile_id_to_index = tile_id_to_index

        def __len__(self):
            return self.inputs.shape[0]

        def __getitem__(self, idx):
            input_data = torch.from_numpy(self.inputs[idx])   # (C,H,W)
            target_data = torch.from_numpy(self.targets[idx]) # (1,H,W)
            time_data = self.times[idx]
            tile_data = self.tile_ids[idx]

            # Fetch the elevation for this tile
            tile_idx = self.tile_id_to_index[tile_data]
            elev_data = torch.from_numpy(self.tile_elev[tile_idx]) # (1,Hf,Wf)

            return input_data, elev_data, target_data, time_data, tile_data

    # Load arrays
    train_input_path = os.path.join(PROCESSED_DIR, "combined_train_input.npy")
    train_target_path = os.path.join(PROCESSED_DIR, "combined_train_target.npy")
    train_times_path = os.path.join(PROCESSED_DIR, "combined_train_times.npy")
    train_tile_ids_path = os.path.join(PROCESSED_DIR, "combined_train_tile_ids.npy")

    test_input_path = os.path.join(PROCESSED_DIR, "combined_test_input.npy")
    test_target_path = os.path.join(PROCESSED_DIR, "combined_test_target.npy")
    test_times_path = os.path.join(PROCESSED_DIR, "combined_test_times.npy")
    test_tile_ids_path = os.path.join(PROCESSED_DIR, "combined_test_tile_ids.npy")

    tile_elev_path = os.path.join(PROCESSED_DIR, "combined_tile_elev.npy")

    train_input = np.load(train_input_path)
    train_target = np.load(train_target_path)
    train_times = np.load(train_times_path)
    train_tile_ids = np.load(train_tile_ids_path)

    test_input = np.load(test_input_path)
    test_target = np.load(test_target_path)
    test_times = np.load(test_times_path)
    test_tile_ids = np.load(test_tile_ids_path)

    tile_elev = np.load(tile_elev_path)  # (num_tiles,1,Hf,Wf)

    # Create a mapping from tile_id -> index in tile_elev
    # Assuming tiles is something like range(0,36), tile_id matches index if tiles are 0-based.
    # If not guaranteed, we can create a mapping:
    unique_tile_ids = sorted(set(train_tile_ids) | set(test_tile_ids))
    tile_id_to_index = {t: i for i, t in enumerate(unique_tile_ids)}

    train_dataset = CombinedDataset(train_input, train_target, train_times, train_tile_ids, tile_elev, tile_id_to_index)
    test_dataset = CombinedDataset(test_input, test_target, test_times, test_tile_ids, tile_elev, tile_id_to_index)

    # Use a fixed seed for deterministic shuffling
    loader_generator = torch.Generator().manual_seed(42)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=loader_generator)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader
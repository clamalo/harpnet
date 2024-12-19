"""
Custom dataset and dataloaders for the HARPNET project.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    """
    Combined dataset for HARPNET including multiple tiles and times.
    """

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        times: np.ndarray,
        tile_ids: np.ndarray,
        tile_elev: np.ndarray,
        tile_id_to_index: dict
    ):
        """
        Args:
            inputs: Numpy array of shape (N, C, H, W)
            targets: Numpy array of shape (N, 1, H, W)
            times: Numpy array of timestamps
            tile_ids: Numpy array of tile IDs (N,)
            tile_elev: Numpy array of shape (num_tiles, 1, Hf, Wf)
            tile_id_to_index: Dictionary mapping tile_id to its index in tile_elev
        """
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
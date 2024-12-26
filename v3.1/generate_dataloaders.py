import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    PROCESSED_DIR,
    BATCH_SIZE
)
from tiles import get_tile_dict, tile_coordinates


class PrecipDataset(Dataset):
    """
    A custom PyTorch Dataset for the precipitation data arrays produced by data_preprocessing.py.

    Each sample contains:
        - 'input'  (coarse-resolution grid),
        - 'target' (fine-resolution grid),
        - 'time'   ([year, month, day, hour]),
        - 'tile'   (tile ID).

    Args:
        inputs (np.ndarray):  Coarse-resolution data array of shape (N, cLat, cLon).
        targets (np.ndarray): Fine-resolution data array of shape (N, fLat, fLon).
        times (np.ndarray):   Timestamp array of shape (N, 4), each row is [year, month, day, hour].
        tiles (np.ndarray):   Tile ID array of shape (N,).
        transform (callable, optional):
            An optional transform function or callable that takes a sample dictionary
            and returns a transformed version.
    """

    def __init__(self, inputs, targets, times, tiles, transform=None):
        self.inputs = inputs
        self.targets = targets
        self.times = times
        self.tiles = tiles
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {
            'input':  self.inputs[idx],
            'target': self.targets[idx],
            'time':   self.times[idx],
            'tile':   self.tiles[idx]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def generate_dataloaders():
    """
    Generate and return training/testing DataLoaders for precipitation data.

    This function:
      1) Loads the compressed .npz file located at PROCESSED_DIR / 'combined_data.npz',
         which should be created by data_preprocessing.py.
      2) Constructs two PrecipDataset instances for training and testing sets.
      3) Wraps them in PyTorch DataLoaders:
         - The training DataLoader is shuffled.
         - The testing DataLoader is not shuffled.
      4) Returns the two DataLoaders as a tuple: (train_loader, test_loader).

    Returns:
        (DataLoader, DataLoader):
            - train_loader: A DataLoader for the training set (shuffled).
            - test_loader:  A DataLoader for the testing set (not shuffled).

    Raises:
        FileNotFoundError: If combined_data.npz is not found in PROCESSED_DIR.
    """
    data_path = PROCESSED_DIR / "combined_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cannot find {data_path}. Make sure you've run data_preprocessing.py first."
        )

    # Load the compressed arrays
    data = np.load(data_path)
    train_input = data['train_input']
    train_target = data['train_target']
    train_time = data['train_time']
    train_tile = data['train_tile']

    test_input = data['test_input']
    test_target = data['test_target']
    test_time = data['test_time']
    test_tile = data['test_tile']

    # Create Datasets
    train_dataset = PrecipDataset(
        inputs=train_input,
        targets=train_target,
        times=train_time,
        tiles=train_tile
    )
    test_dataset = PrecipDataset(
        inputs=test_input,
        targets=test_target,
        times=test_time,
        tiles=test_tile
    )

    # Wrap datasets in DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = generate_dataloaders()
    print(f"Training DataLoader: {len(train_loader)} batches")
    print(f"Testing DataLoader: {len(test_loader)} batches")
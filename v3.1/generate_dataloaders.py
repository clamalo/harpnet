import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from config import (
    PROCESSED_DIR,
    BATCH_SIZE
)


class PrecipDataset(Dataset):
    """
    A custom PyTorch Dataset for the precipitation data arrays produced by data_preprocessing.py.

    Each sample contains:
        - 'input':  (2, fLat, fLon), which is [upsampled_coarse_precip, fine_elevation].
        - 'coarse_input': (cLat, cLon) - the original coarse-resolution precipitation array
          (no padding removed here, but no upsampling).
        - 'target': (fLat, fLon) - the high-resolution ground truth.
        - 'time':   (4,)  [year, month, day, hour].
        - 'tile':   (int) the tile ID.

    The upsampling to fine resolution (for the model's first channel) is done on-the-fly.
    Meanwhile, we now explicitly return the coarse array so that we can later
    perform a correct bilinear interpolation baseline that reflects the tile's
    true bounding box, without confusion from any padding.
    """

    def __init__(self, inputs, targets, times, tiles,
                 tile_elevations=None, tile_ids=None,
                 transform=None):
        """
        Args:
            inputs (np.ndarray):  Coarse-resolution data array of shape (N, cLat, cLon).
                                  This is log(1+precip) normalized if configured by preprocessing.
            targets (np.ndarray): Fine-resolution data array of shape (N, fLat, fLon).
                                  Also log(1+precip) normalized if configured by preprocessing.
            times (np.ndarray):   (N, 4), [year, month, day, hour].
            tiles (np.ndarray):   (N,).
            tile_elevations (np.ndarray): (num_tiles, fLat, fLon). Elevation for each tile ID.
            tile_ids (np.ndarray): 1D array of shape (num_tiles,) matching tile IDs to the
                                   index in tile_elevations.
            transform (callable, optional): optional transform for augmentation.
        """
        self.inputs = inputs
        self.targets = targets
        self.times = times
        self.tiles = tiles
        self.transform = transform

        # Build a lookup dict for tile ID -> index into tile_elevations
        self.tile_elevations = tile_elevations
        self.tile_id_to_index = {}
        if tile_elevations is not None and tile_ids is not None:
            for i, tid in enumerate(tile_ids):
                self.tile_id_to_index[tid] = i

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Returns a dict:
            {
              'input':        (2, fLat, fLon),
              'coarse_input': (cLat, cLon),
              'target':       (fLat, fLon),
              'time':         (4,),
              'tile':         <tile_id>
            }
        """
        # (cLat, cLon) coarse precipitation
        coarse_precip = self.inputs[idx]  # [already normalized log-space]

        # (fLat, fLon) fine-resolution target
        target = self.targets[idx]

        # Identify tile & retrieve its fine-res elevation
        tile_id = self.tiles[idx]
        elev = None
        if self.tile_elevations is not None:
            tile_idx = self.tile_id_to_index.get(tile_id, None)
            if tile_idx is not None:
                elev = self.tile_elevations[tile_idx]  # shape (fLat, fLon)
        if elev is None:
            elev = np.zeros_like(target)

        # Upsample coarse_precip -> (fLat, fLon) for the model's input channel
        coarse_precip_t = torch.from_numpy(coarse_precip).unsqueeze(0).unsqueeze(0)
        fLat, fLon = elev.shape
        upsampled_precip_t = F.interpolate(
            coarse_precip_t,
            size=(fLat, fLon),
            mode='bilinear',
            align_corners=False
        )
        upsampled_precip = upsampled_precip_t.squeeze().numpy()

        # Stack [precip, elevation]
        combined_input = np.stack([upsampled_precip, elev], axis=0)

        sample = {
            'input':        combined_input,   # (2, fLat, fLon)
            'coarse_input': coarse_precip,    # (cLat, cLon)
            'target':       target,           # (fLat, fLon)
            'time':         self.times[idx],  # (4,)
            'tile':         tile_id
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def generate_dataloaders(tile_id=None):
    """
    Generate and return training/testing DataLoaders for precipitation data plus elevation.

    If tile_id is None, this function returns DataLoaders for the entire dataset
    (the original train/test split). If tile_id is specified, it filters the dataset
    so only samples matching that tile_id are included in the returned train/test sets.

    Returns:
        (train_loader, test_loader)

    Each sample from these loaders will now contain:
        - 'input':        (2, fLat, fLon)  [upsampled coarse precip + fine elev]
        - 'coarse_input': (cLat, cLon)     [coarse data without upsampling]
        - 'target':       (fLat, fLon)
        - 'time':         (4,)
        - 'tile':         tile_id
    """
    data_path = PROCESSED_DIR / "combined_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cannot find {data_path}. Make sure you've run data_preprocessing.py first."
        )

    # Load the compressed arrays
    data = np.load(data_path)
    train_input = data['train_input']    # shape (N_train, cLat, cLon)
    train_target = data['train_target']  # shape (N_train, fLat, fLon)
    train_time = data['train_time']
    train_tile = data['train_tile']

    test_input = data['test_input']      # shape (N_test, cLat, cLon)
    test_target = data['test_target']    # shape (N_test, fLat, fLon)
    test_time = data['test_time']
    test_tile = data['test_tile']

    tile_elevations = data.get('tile_elevations', None)  # (num_tiles, fLat, fLon)
    tile_ids = data.get('tile_ids', None)
    data.close()

    # Optionally filter by tile_id
    if tile_id is not None:
        train_mask = (train_tile == tile_id)
        test_mask = (test_tile == tile_id)

        train_input = train_input[train_mask]
        train_target = train_target[train_mask]
        train_time = train_time[train_mask]
        train_tile = train_tile[train_mask]

        test_input = test_input[test_mask]
        test_target = test_target[test_mask]
        test_time = test_time[test_mask]
        test_tile = test_tile[test_mask]

    # Create Datasets
    train_dataset = PrecipDataset(
        inputs=train_input,
        targets=train_target,
        times=train_time,
        tiles=train_tile,
        tile_elevations=tile_elevations,
        tile_ids=tile_ids
    )
    test_dataset = PrecipDataset(
        inputs=test_input,
        targets=test_target,
        times=test_time,
        tiles=test_tile,
        tile_elevations=tile_elevations,
        tile_ids=tile_ids
    )

    g = torch.Generator()
    g.manual_seed(42)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        generator=g
    )

    return train_loader, test_loader
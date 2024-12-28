import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from config import (
    PROCESSED_DIR,
    BATCH_SIZE,
    COARSE_RESOLUTION,
    FINE_RESOLUTION,
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

    The upsampling to fine resolution (for the model's first channel) is now done with a 
    "bigger interpolation + center crop" approach, so that the final precipitation grid 
    exactly matches the resolution and spatial extent of the 'target' after removing the 
    outer pad region.
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
        # (cLat, cLon) coarse precipitation (normalized log-space)
        coarse_precip = self.inputs[idx]

        # (fLat, fLon) fine-resolution target (normalized log-space)
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

        # Upsample with padding -> bigger shape -> then crop to exactly match target shape
        upsampled_precip = _upsample_coarse_with_crop(
            coarse_array=coarse_precip,
            final_shape=target.shape  # e.g. (fLat, fLon)
        )

        # Stack [precip, elevation] as 2 channels
        combined_input = np.stack([upsampled_precip, elev], axis=0)

        sample = {
            'input':        combined_input,   # (2, fLat, fLon)
            'coarse_input': coarse_precip,    # (cLat, cLon) original coarse data
            'target':       target,           # (fLat, fLon)
            'time':         self.times[idx],  # (4,)
            'tile':         tile_id
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def _upsample_coarse_with_crop(coarse_array: np.ndarray, final_shape: tuple) -> np.ndarray:
    """
    Interpolate a coarse precipitation array to a larger padded size, then
    center-crop it so that the final shape matches 'final_shape'.

    For example, if coarse_array is 18×18 and final_shape is 64×64, we might:
      1) upsample to 72×72 (i.e. 4× upsampling for 18×18).
      2) center-crop out 64×64 from the middle, discarding 4 px on each edge.

    Args:
        coarse_array (np.ndarray): shape (cLat, cLon), log-space precipitation.
        final_shape (tuple): (fLat, fLon) shape for the final cropped result.

    Returns:
        np.ndarray of shape (fLat, fLon)
    """
    cLat, cLon = coarse_array.shape
    fLat, fLon = final_shape

    # Compute integer upsampling factor (assuming coarse/fine ratio is consistent in lat/lon)
    ratio = int(round(COARSE_RESOLUTION / FINE_RESOLUTION))

    upsample_size_lat = cLat * ratio
    upsample_size_lon = cLon * ratio

    if upsample_size_lat < fLat or upsample_size_lon < fLon:
        raise ValueError(
            f"Upsample size ({upsample_size_lat}×{upsample_size_lon}) is smaller "
            f"than final shape ({fLat}×{fLon}). Check your config!"
        )

    # Step 1) Interpolate to (upsample_size_lat, upsample_size_lon)
    coarse_tensor = torch.from_numpy(coarse_array).unsqueeze(0).unsqueeze(0).float()
    upsampled_t = F.interpolate(
        coarse_tensor,
        size=(upsample_size_lat, upsample_size_lon),
        mode='bilinear',
        align_corners=False
    )

    # Step 2) Center-crop to (fLat, fLon)
    lat_diff = upsample_size_lat - fLat
    lon_diff = upsample_size_lon - fLon

    top = lat_diff // 2
    left = lon_diff // 2
    bottom = top + fLat
    right = left + fLon

    cropped_t = upsampled_t[:, :, top:bottom, left:right]

    # Return as np.ndarray (fLat, fLon)
    return cropped_t.squeeze(0).squeeze(0).numpy()


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

    # Load the compressed arrays (metadata + references)
    data = np.load(data_path, allow_pickle=True)

    # Check if we have memmap references in the .npz
    if 'train_input_mm_filename' in data:
        train_input_mm_filename = data['train_input_mm_filename'].item()
        train_input_shape = tuple(data['train_input_shape'])
        train_input = np.memmap(train_input_mm_filename, dtype='float16', mode='r', shape=train_input_shape)

        train_target_mm_filename = data['train_target_mm_filename'].item()
        train_target_shape = tuple(data['train_target_shape'])
        train_target = np.memmap(train_target_mm_filename, dtype='float16', mode='r', shape=train_target_shape)

        test_input_mm_filename = data['test_input_mm_filename'].item()
        test_input_shape = tuple(data['test_input_shape'])
        test_input = np.memmap(test_input_mm_filename, dtype='float16', mode='r', shape=test_input_shape)

        test_target_mm_filename = data['test_target_mm_filename'].item()
        test_target_shape = tuple(data['test_target_shape'])
        test_target = np.memmap(test_target_mm_filename, dtype='float16', mode='r', shape=test_target_shape)

    else:
        # Fallback in case the user never used memmaps
        # (Not really expected if you are using the updated data_preprocessing.py,
        # but we keep it for backward compatibility.)
        train_input = data['train_input']
        train_target = data['train_target']
        test_input = data['test_input']
        test_target = data['test_target']

    train_time = data['train_time']
    test_time = data['test_time']
    train_tile = data['train_tile']
    test_tile = data['test_tile']

    tile_elevations = data.get('tile_elevations', None)
    tile_ids = data.get('tile_ids', None)
    data.close()

    # Optionally filter by tile_id
    if tile_id is not None:
        train_mask = (train_tile == tile_id)
        test_mask = (test_tile == tile_id)

        # We must take care if the arrays are memmaps, we can slice them using mask 
        # but that may produce an in-memory copy. This is acceptable for small subsets
        # but be aware it can still be large. Usually tile-based subsets are smaller.
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
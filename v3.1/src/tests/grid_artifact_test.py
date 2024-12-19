"""
This test script checks for grid artifacts by evaluating model outputs vs bilinear interpolation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm

from src.data.dataloaders import generate_dataloaders
from src.config import Config
from src.models.model import UNetWithAttention
from src.utils.get_coordinates import get_coordinates
from src.utils.setup import setup
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

# Example usage parameters:
start_month = (1979, 10)
end_month = (1980, 9)
train_test_ratio = 0.2
tile = 31

setup()

checkpoint_path = Config.CHECKPOINTS_DIR / f"{tile}_model.pt"
if not checkpoint_path.exists():
    logger.error(f"Checkpoint for tile {tile} not found. Please ensure it's available.")
else:
    model = UNetWithAttention().to(Config.TORCH_DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=Config.TORCH_DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    coarse_lats_pad, coarse_lons_pad, coarse_lats, coarse_lons, fine_lats, fine_lons = get_coordinates(tile)

    train_dataloader, test_dataloader = generate_dataloaders([tile], start_month, end_month, train_test_ratio)

    losses = []
    bilinear_losses = []

    model.eval()
    with torch.no_grad():
        for (inputs, elev_data, targets, times, tile_ids) in tqdm(test_dataloader, total=len(test_dataloader), desc="Testing Grid Artifact"):
            inputs, elev_data, targets = inputs.to(Config.TORCH_DEVICE), elev_data.to(Config.TORCH_DEVICE), targets.to(Config.TORCH_DEVICE)

            inputs_resized = torch.nn.functional.interpolate(inputs, size=(64,64), mode='nearest')
            elev_data = torch.nn.functional.interpolate(elev_data, size=(64,64), mode='nearest')
            inputs_combined = torch.cat([inputs_resized, elev_data], dim=1)
            outputs = model(inputs_combined)

            loss = ((outputs - targets) ** 2).cpu().numpy()
            cropped_inputs = inputs_resized[:,0:1,1:-1,1:-1]
            interpolated_inputs = torch.nn.functional.interpolate(cropped_inputs, size=(64, 64), mode='bilinear')
            bilinear_loss = ((interpolated_inputs - targets) ** 2).cpu().numpy()

            losses.append(loss.mean(axis=0))
            bilinear_losses.append(bilinear_loss.mean(axis=0))

    mean_losses = np.array(losses).mean(axis=0)
    mean_bilinear_losses = np.array(bilinear_losses).mean(axis=0)
    max_value = max(mean_losses.max(), mean_bilinear_losses.max())

    # Plot model losses
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    ax.add_feature(cfeature.STATES)
    ax.set_extent([fine_lons[0], fine_lons[-1], fine_lats[0], fine_lats[-1]])
    ax.set_title('Mean Losses (Model)')
    cf = ax.imshow(mean_losses, origin='upper', extent=(fine_lons[0], fine_lons[-1], fine_lats[0], fine_lats[-1]),
                   transform=ccrs.PlateCarree(), vmin=0, vmax=max_value)
    plt.colorbar(cf, ax=ax, orientation='horizontal', label='Mean Losses')
    for lat in coarse_lats:
        ax.axhline(y=lat, color='red', linestyle='--', linewidth=0.5)
    for lon in coarse_lons:
        ax.axvline(x=lon, color='red', linestyle='--', linewidth=0.5)
    plt.savefig('model_losses.png')
    plt.close(fig)

    # Plot bilinear losses
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    ax.add_feature(cfeature.STATES)
    ax.set_extent([fine_lons[0], fine_lons[-1], fine_lats[0], fine_lats[-1]])
    ax.set_title('Mean Losses (Bilinear)')
    cf = ax.imshow(mean_bilinear_losses, origin='upper', extent=(fine_lons[0], fine_lons[-1], fine_lats[0], fine_lats[-1]),
                   transform=ccrs.PlateCarree(), vmin=0, vmax=max_value)
    plt.colorbar(cf, ax=ax, orientation='horizontal', label='Mean Losses')
    for lat in coarse_lats:
        ax.axhline(y=lat, color='red', linestyle='--', linewidth=0.5)
    for lon in coarse_lons:
        ax.axvline(x=lon, color='red', linestyle='--', linewidth=0.5)
    plt.savefig('bilinear_losses.png')
    plt.close(fig)

    logger.info("Grid artifact test completed. Check model_losses.png and bilinear_losses.png.")
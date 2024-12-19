"""
Training logic for the UNetWithAttention model.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional
from src.config import Config
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def train_one_epoch(
    model: nn.Module, 
    train_dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module,
    device: str
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model.
        train_dataloader: DataLoader for training.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device string.

    Returns:
        float: Average training loss.
    """
    model.train()
    train_losses = []

    for batch in tqdm(train_dataloader, desc="Training", unit="batch"):
        inputs, elev_data, targets, times, tile_ids = batch

        inputs = torch.nn.functional.interpolate(inputs, size=(64, 64), mode='nearest')
        elev_data = torch.nn.functional.interpolate(elev_data, size=(64,64), mode='nearest')
        inputs = torch.cat([inputs, elev_data], dim=1)

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    avg_loss = sum(train_losses) / len(train_losses) if train_losses else float('inf')
    return avg_loss


def test_model(
    model: nn.Module,
    test_dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    focus_tile: Optional[int] = None
) -> (float, float, Optional[float]):
    """
    Test the model on the test_dataloader and compute test_loss and bilinear_test_loss.
    If focus_tile is provided, also compute tile-specific test loss.

    Returns:
        (test_loss, bilinear_test_loss, focus_tile_test_loss)
    """
    model.eval()
    test_losses = []
    bilinear_test_losses = []
    focus_tile_losses = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing", unit="batch"):
            inputs, elev_data, targets, times, tile_ids = batch

            inputs = torch.nn.functional.interpolate(inputs, size=(64, 64), mode='nearest')
            elev_data = torch.nn.functional.interpolate(elev_data, size=(64, 64), mode='nearest')
            inputs = torch.cat([inputs, elev_data], dim=1)

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Compute bilinear baseline loss
            cropped_inputs = inputs[:,0:1,1:-1,1:-1]
            interpolated_inputs = torch.nn.functional.interpolate(cropped_inputs, size=(64, 64), mode='bilinear')
            bilinear_loss = criterion(interpolated_inputs, targets)

            test_losses.append(loss.item())
            bilinear_test_losses.append(bilinear_loss.item())

            if focus_tile is not None:
                mask = (tile_ids == focus_tile)
                if mask.any():
                    mask_indices = torch.where(mask)[0].to(device)
                    focus_outputs = outputs[mask_indices]
                    focus_targets = targets[mask_indices]
                    focus_loss = criterion(focus_outputs, focus_targets)
                    focus_tile_losses.append(focus_loss.item())

    mean_test_loss = sum(test_losses) / len(test_losses) if test_losses else float('inf')
    mean_bilinear_loss = sum(bilinear_test_losses) / len(bilinear_test_losses) if bilinear_test_losses else float('inf')

    if focus_tile is not None and focus_tile_losses:
        focus_tile_test_loss = sum(focus_tile_losses) / len(focus_tile_losses)
    else:
        focus_tile_test_loss = None

    return mean_test_loss, mean_bilinear_loss, focus_tile_test_loss


def train_test(
    train_dataloader: DataLoader, 
    test_dataloader: DataLoader, 
    start_epoch: int=0, 
    end_epoch: int=20, 
    focus_tile: Optional[int]=None
):
    """
    Train and test the model for a given number of epochs.
    Saves a checkpoint after each epoch.
    Optionally compute loss specific to a focus_tile.
    """
    random = torch.manual_seed(Config.RANDOM_SEED)
    np_random = torch.manual_seed(Config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    from src.models.model import UNetWithAttention
    device = Config.TORCH_DEVICE

    model = UNetWithAttention().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    if start_epoch != 0:
        checkpoint_path = Config.CHECKPOINTS_DIR / f'{start_epoch-1}_model.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(start_epoch, end_epoch):
        logger.info(f"Epoch {epoch} starting...")
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
        test_loss, bilinear_test_loss, focus_tile_loss = test_model(model, test_dataloader, criterion, device, focus_tile)

        logger.info(f'Epoch {epoch}: Train={train_loss:.4f}, Test={test_loss:.4f}, Bilinear={bilinear_test_loss:.4f}')
        if focus_tile is not None:
            if focus_tile_loss is not None:
                logger.info(f"Epoch {epoch}: Focus Tile {focus_tile} Loss={focus_tile_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch}: No samples found for tile {focus_tile} in test set.")

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'bilinear_test_loss': bilinear_test_loss
        }, Config.CHECKPOINTS_DIR / f'{epoch}_model.pt')
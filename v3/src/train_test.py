"""
Training and testing routines for the model.
Includes functions for one epoch of training and testing, and a full train-test loop.

Now updated so that normalization stats are loaded lazily, i.e., only when test_model()
actually needs them. This prevents issues where the file normalization_stats.npy might
not exist until after data processing is completed.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import random
import numpy as np
import logging
from typing import Optional
from src.constants import (
    TORCH_DEVICE,
    CHECKPOINTS_DIR,
    RANDOM_SEED,
    MODEL_NAME,
    NORMALIZATION_STATS_FILE,
    TILE_SIZE
)
from src.generate_dataloaders import generate_dataloaders
import importlib
from tabulate import tabulate

# Dynamically import model based on MODEL_NAME
model_module = importlib.import_module(f"src.models.{MODEL_NAME}")
ModelClass = model_module.Model

def _lazy_load_stats():
    """
    Loads normalization stats only when this function is called.
    This prevents FileNotFoundError if the file is not yet generated when 'train_test.py' is imported.
    """
    norm_stats = np.load(NORMALIZATION_STATS_FILE)  # Raises an error if file not found
    mean_val, std_val = float(norm_stats[0]), float(norm_stats[1])
    return mean_val, std_val


def train_one_epoch(model: nn.Module, 
                    train_dataloader, 
                    optimizer: torch.optim.Optimizer, 
                    criterion: nn.Module) -> float:
    """
    Train the model for one epoch (using normalized data).

    Args:
        model: The model to train.
        train_dataloader: DataLoader for training data (normalized).
        optimizer: Torch optimizer.
        criterion: Loss function (MSE in normalized space).

    Returns:
        Average training loss over the epoch (in normalized space).
    """
    model.train()
    train_losses = []

    for batch in tqdm(train_dataloader, desc="Training", unit="batch"):
        inputs, elev_data, targets, times, tile_ids = batch
        inputs = inputs.to(TORCH_DEVICE)
        targets = targets.to(TORCH_DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)  # Normalized MSE loss
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    return sum(train_losses) / len(train_losses) if train_losses else float('inf')


def test_model(model: nn.Module, 
               test_dataloader, 
               criterion: nn.Module, 
               focus_tile: Optional[int] = None):
    """
    Test the model on the test dataset in normalized space, but also compute metrics in unnormalized space.

    Args:
        model: The trained model (expects normalized inputs).
        test_dataloader: DataLoader for testing data (normalized).
        criterion: Typically MSELoss in normalized space.
        focus_tile: Optional tile ID to compute metrics for that specific tile as well.

    Returns:
        A dictionary containing all normalized and unnormalized metrics.
    """
    # Load mean/std stats at the time of testing
    MEAN_VAL, STD_VAL = _lazy_load_stats()

    model.eval()
    test_losses = []
    bilinear_test_losses = []
    focus_tile_losses = []
    focus_tile_bilinear_losses = []

    # For additional unnormalized metrics
    all_preds_norm = []
    all_targets_norm = []
    all_bilinear_norm = []

    focus_tile_preds_norm = []
    focus_tile_targets_norm = []
    focus_tile_bilinear_norm = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing", unit="batch"):
            inputs, elev_data, targets, times, tile_ids = batch
            inputs = inputs.to(TORCH_DEVICE)
            targets = targets.to(TORCH_DEVICE)
            outputs = model(inputs)  # normalized predictions

            # MSE in normalized space
            loss = criterion(outputs, targets)

            # Bilinear baseline in normalized space (nearest from cropped input)
            cropped_inputs = inputs[:, 0:1, 1:-1, 1:-1]
            interpolated_inputs = torch.nn.functional.interpolate(
                cropped_inputs, size=(TILE_SIZE, TILE_SIZE), mode='bilinear'
            )
            bilinear_loss = criterion(interpolated_inputs, targets)

            test_losses.append(loss.item())
            bilinear_test_losses.append(bilinear_loss.item())

            # Store normalized predictions for unnormalizing later
            all_preds_norm.append(outputs.cpu().numpy())
            all_targets_norm.append(targets.cpu().numpy())
            all_bilinear_norm.append(interpolated_inputs.cpu().numpy())

            if focus_tile is not None:
                mask = (tile_ids == focus_tile)
                if mask.any():
                    focus_outputs_norm = outputs[mask.to(TORCH_DEVICE)]
                    focus_targets_norm = targets[mask.to(TORCH_DEVICE)]
                    focus_bilinear_norm_ = interpolated_inputs[mask.to(TORCH_DEVICE)]

                    focus_loss = criterion(focus_outputs_norm, focus_targets_norm)
                    focus_tile_losses.append(focus_loss.item())

                    focus_bilinear_loss_ = criterion(focus_bilinear_norm_, focus_targets_norm)
                    focus_tile_bilinear_losses.append(focus_bilinear_loss_.item())

                    focus_tile_preds_norm.append(focus_outputs_norm.cpu().numpy())
                    focus_tile_targets_norm.append(focus_targets_norm.cpu().numpy())
                    focus_tile_bilinear_norm.append(focus_bilinear_norm_.cpu().numpy())

    mean_test_loss = sum(test_losses) / len(test_losses) if test_losses else float('inf')
    mean_bilinear_loss = sum(bilinear_test_losses) / len(bilinear_test_losses) if bilinear_test_losses else float('inf')

    if focus_tile is not None and focus_tile_losses:
        focus_tile_test_loss = sum(focus_tile_losses) / len(focus_tile_losses)
        focus_tile_bilinear_loss = sum(focus_tile_bilinear_losses) / len(focus_tile_bilinear_losses)
    else:
        focus_tile_test_loss = None
        focus_tile_bilinear_loss = None

    # Unnormalize for additional metrics
    all_preds_norm = np.concatenate(all_preds_norm, axis=0)
    all_targets_norm = np.concatenate(all_targets_norm, axis=0)
    all_bilinear_norm = np.concatenate(all_bilinear_norm, axis=0)

    all_preds_unnorm = (all_preds_norm * STD_VAL) + MEAN_VAL
    all_targets_unnorm = (all_targets_norm * STD_VAL) + MEAN_VAL
    all_bilinear_unnorm = (all_bilinear_norm * STD_VAL) + MEAN_VAL

    preds_unnorm_flat = all_preds_unnorm.flatten()
    targets_unnorm_flat = all_targets_unnorm.flatten()
    bilinear_unnorm_flat = all_bilinear_unnorm.flatten()

    unnorm_test_mse = np.mean((preds_unnorm_flat - targets_unnorm_flat)**2)
    unnorm_bilinear_mse = np.mean((bilinear_unnorm_flat - targets_unnorm_flat)**2)

    unnorm_test_mae = np.mean(np.abs(preds_unnorm_flat - targets_unnorm_flat))
    unnorm_bilinear_mae = np.mean(np.abs(bilinear_unnorm_flat - targets_unnorm_flat))

    if np.std(targets_unnorm_flat) > 1e-8 and np.std(preds_unnorm_flat) > 1e-8:
        unnorm_test_corr = np.corrcoef(targets_unnorm_flat, preds_unnorm_flat)[0, 1]
    else:
        unnorm_test_corr = float('nan')

    if np.std(targets_unnorm_flat) > 1e-8 and np.std(bilinear_unnorm_flat) > 1e-8:
        unnorm_bilinear_corr = np.corrcoef(targets_unnorm_flat, bilinear_unnorm_flat)[0, 1]
    else:
        unnorm_bilinear_corr = float('nan')

    if focus_tile is not None and focus_tile_losses:
        ft_preds_norm = np.concatenate(focus_tile_preds_norm, axis=0)
        ft_targets_norm = np.concatenate(focus_tile_targets_norm, axis=0)
        ft_bilinear_norm = np.concatenate(focus_tile_bilinear_norm, axis=0)

        ft_preds_unnorm = (ft_preds_norm * STD_VAL) + MEAN_VAL
        ft_targets_unnorm = (ft_targets_norm * STD_VAL) + MEAN_VAL
        ft_bilinear_unnorm = (ft_bilinear_norm * STD_VAL) + MEAN_VAL

        ft_preds_unnorm_flat = ft_preds_unnorm.flatten()
        ft_targets_unnorm_flat = ft_targets_unnorm.flatten()
        ft_bilinear_unnorm_flat = ft_bilinear_unnorm.flatten()

        unnorm_focus_tile_mse = np.mean((ft_preds_unnorm_flat - ft_targets_unnorm_flat)**2)
        unnorm_focus_tile_bilinear_mse = np.mean((ft_bilinear_unnorm_flat - ft_targets_unnorm_flat)**2)

        unnorm_focus_tile_mae = np.mean(np.abs(ft_preds_unnorm_flat - ft_targets_unnorm_flat))
        unnorm_focus_tile_bilinear_mae = np.mean(np.abs(ft_bilinear_unnorm_flat - ft_targets_unnorm_flat))

        if np.std(ft_targets_unnorm_flat) > 1e-8 and np.std(ft_preds_unnorm_flat) > 1e-8:
            unnorm_focus_tile_corr = np.corrcoef(ft_targets_unnorm_flat, ft_preds_unnorm_flat)[0, 1]
        else:
            unnorm_focus_tile_corr = float('nan')

        if np.std(ft_targets_unnorm_flat) > 1e-8 and np.std(ft_bilinear_unnorm_flat) > 1e-8:
            unnorm_focus_tile_bilinear_corr = np.corrcoef(ft_targets_unnorm_flat, ft_bilinear_unnorm_flat)[0, 1]
        else:
            unnorm_focus_tile_bilinear_corr = float('nan')
    else:
        unnorm_focus_tile_mse = None
        unnorm_focus_tile_bilinear_mse = None
        unnorm_focus_tile_mae = None
        unnorm_focus_tile_bilinear_mae = None
        unnorm_focus_tile_corr = None
        unnorm_focus_tile_bilinear_corr = None

    return {
        "mean_test_loss": mean_test_loss,
        "mean_bilinear_loss": mean_bilinear_loss,
        "focus_tile_test_loss": focus_tile_test_loss,
        "focus_tile_bilinear_loss": focus_tile_bilinear_loss,
        "unnorm_test_mse": unnorm_test_mse,
        "unnorm_bilinear_mse": unnorm_bilinear_mse,
        "unnorm_focus_tile_mse": unnorm_focus_tile_mse,
        "unnorm_focus_tile_bilinear_mse": unnorm_focus_tile_bilinear_mse,
        "unnorm_test_mae": unnorm_test_mae,
        "unnorm_bilinear_mae": unnorm_bilinear_mae,
        "unnorm_focus_tile_mae": unnorm_focus_tile_mae,
        "unnorm_focus_tile_bilinear_mae": unnorm_focus_tile_bilinear_mae,
        "unnorm_test_corr": unnorm_test_corr,
        "unnorm_bilinear_corr": unnorm_bilinear_corr,
        "unnorm_focus_tile_corr": unnorm_focus_tile_corr,
        "unnorm_focus_tile_bilinear_corr": unnorm_focus_tile_bilinear_corr
    }


def train_test(train_dataloader, 
               test_dataloader, 
               start_epoch: int = 0, 
               end_epoch: int = 20, 
               focus_tile: Optional[int] = None):
    """
    Full training/testing routine for a given model architecture.
    Saves checkpoints at each epoch.

    After each epoch, metrics are displayed in a nicely formatted table for easier analysis.
    """

    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = ModelClass().to(TORCH_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()  # Normalized MSE

    # If resuming, load previous checkpoint
    if start_epoch != 0:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'{start_epoch-1}_model.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=TORCH_DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(start_epoch, end_epoch):
        logging.info(f"\nEpoch {epoch} starting...")
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion)

        metrics = test_model(model, test_dataloader, criterion, focus_tile)

        mean_test_loss = metrics["mean_test_loss"]
        mean_bilinear_loss = metrics["mean_bilinear_loss"]
        focus_tile_test_loss = metrics["focus_tile_test_loss"]
        unnorm_test_mse = metrics["unnorm_test_mse"]
        unnorm_bilinear_mse = metrics["unnorm_bilinear_mse"]
        unnorm_focus_tile_mse = metrics["unnorm_focus_tile_mse"]
        unnorm_test_mae = metrics["unnorm_test_mae"]
        unnorm_bilinear_mae = metrics["unnorm_bilinear_mae"]
        unnorm_focus_tile_mae = metrics["unnorm_focus_tile_mae"]
        unnorm_test_corr = metrics["unnorm_test_corr"]
        unnorm_bilinear_corr = metrics["unnorm_bilinear_corr"]
        unnorm_focus_tile_corr = metrics["unnorm_focus_tile_corr"]

        ft_mse_norm = f"{focus_tile_test_loss:.6f}" if focus_tile_test_loss is not None else "N/A"
        ft_mse_unnorm = f"{unnorm_focus_tile_mse:.6f}" if unnorm_focus_tile_mse is not None else "N/A"
        ft_mae_unnorm = f"{unnorm_focus_tile_mae:.6f}" if unnorm_focus_tile_mae is not None else "N/A"
        ft_corr_unnorm = f"{unnorm_focus_tile_corr:.4f}" if unnorm_focus_tile_corr is not None else "N/A"

        headers = ["Metric", "Train", "Test", "Bilinear", f"Focus Tile {focus_tile if focus_tile is not None else ''}"]
        data = [
            ["MSE (Normalized)", f"{train_loss:.6f}", f"{mean_test_loss:.6f}", f"{mean_bilinear_loss:.6f}", ft_mse_norm],
            ["MSE (Unnorm)", "-", f"{unnorm_test_mse:.6f}", f"{unnorm_bilinear_mse:.6f}", ft_mse_unnorm],
            ["MAE (Unnorm)", "-", f"{unnorm_test_mae:.6f}", f"{unnorm_bilinear_mae:.6f}", ft_mae_unnorm],
            ["Corr (Unnorm)", "-", f"{unnorm_test_corr:.4f}", f"{unnorm_bilinear_corr:.4f}", ft_corr_unnorm],
        ]
        table_str = tabulate(data, headers=headers, tablefmt="pretty")
        logging.info("\n" + table_str)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': mean_test_loss,
            'bilinear_test_loss': mean_bilinear_loss
        }, os.path.join(CHECKPOINTS_DIR, f'{epoch}_model.pt'))
"""
Training and testing routines for the model.
Includes functions for one epoch of training and testing, and a full train-test loop.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import random
import numpy as np
from typing import Optional
from src.constants import TORCH_DEVICE, CHECKPOINTS_DIR, RANDOM_SEED, MODEL_NAME
from src.generate_dataloaders import generate_dataloaders
import importlib

# Dynamically import model based on MODEL_NAME
model_module = importlib.import_module(f"src.models.{MODEL_NAME}")
ModelClass = model_module.Model

def train_model(model: nn.Module, 
                train_dataloader, 
                optimizer: torch.optim.Optimizer, 
                criterion: nn.Module) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The model to train.
        train_dataloader: DataLoader for training data.
        optimizer: Torch optimizer.
        criterion: Loss function.

    Returns:
        Average training loss over the epoch.
    """
    model.train()
    train_losses = []

    for batch in tqdm(train_dataloader, desc="Training", unit="batch"):
        inputs, elev_data, targets, times, tile_ids = batch
        # inputs already contains elevation as second channel
        inputs = inputs.to(TORCH_DEVICE)
        targets = targets.to(TORCH_DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
    return sum(train_losses) / len(train_losses) if train_losses else float('inf')

def test_model(model: nn.Module, 
               test_dataloader, 
               criterion: nn.Module, 
               focus_tile: Optional[int]=None) -> (float, float, Optional[float], Optional[float]):
    """
    Test the model on the test dataset and also compute bilinear baseline for comparison.
    If focus_tile is provided, also compute losses specifically for that tile.

    Returns:
        mean_test_loss: Average MSE test loss over all test samples.
        mean_bilinear_loss: Average MSE bilinear baseline loss over all test samples.
        focus_tile_test_loss: Average MSE test loss for the focus_tile if provided.
        focus_tile_bilinear_loss: Average MSE bilinear baseline loss for the focus_tile if provided.
    """
    model.eval()
    test_losses = []
    bilinear_test_losses = []
    focus_tile_losses = []
    focus_tile_bilinear_losses = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing", unit="batch"):
            inputs, elev_data, targets, times, tile_ids = batch
            inputs = inputs.to(TORCH_DEVICE)
            targets = targets.to(TORCH_DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Compute bilinear baseline loss for comparison
            cropped_inputs = inputs[:,0:1,1:-1,1:-1]
            interpolated_inputs = torch.nn.functional.interpolate(cropped_inputs, size=(64, 64), mode='bilinear')
            bilinear_loss = criterion(interpolated_inputs, targets)

            test_losses.append(loss.item())
            bilinear_test_losses.append(bilinear_loss.item())

            if focus_tile is not None:
                mask = (tile_ids == focus_tile)
                if mask.any():
                    # Focus tile model loss
                    focus_outputs = outputs[mask.to(TORCH_DEVICE)]
                    focus_targets = targets[mask.to(TORCH_DEVICE)]
                    focus_loss = criterion(focus_outputs, focus_targets)
                    focus_tile_losses.append(focus_loss.item())

                    # Focus tile bilinear loss
                    focus_bilinear_inputs = interpolated_inputs[mask.to(TORCH_DEVICE)]
                    focus_bilinear_loss = criterion(focus_bilinear_inputs, focus_targets)
                    focus_tile_bilinear_losses.append(focus_bilinear_loss.item())

    mean_test_loss = sum(test_losses) / len(test_losses) if test_losses else float('inf')
    mean_bilinear_loss = sum(bilinear_test_losses) / len(bilinear_test_losses) if bilinear_test_losses else float('inf')

    if focus_tile is not None and focus_tile_losses:
        focus_tile_test_loss = sum(focus_tile_losses) / len(focus_tile_losses)
        focus_tile_bilinear_loss = sum(focus_tile_bilinear_losses) / len(focus_tile_bilinear_losses) if focus_tile_bilinear_losses else None
    else:
        focus_tile_test_loss = None
        focus_tile_bilinear_loss = None

    return mean_test_loss, mean_bilinear_loss, focus_tile_test_loss, focus_tile_bilinear_loss

def train_test(train_dataloader, 
               test_dataloader, 
               start_epoch: int=0, 
               end_epoch: int=20, 
               focus_tile: Optional[int]=None):
    """
    Full training/testing routine for a given model architecture.
    Saves checkpoints at each epoch.

    Args:
        train_dataloader: DataLoader for training.
        test_dataloader: DataLoader for testing.
        start_epoch: Starting epoch (useful for resuming).
        end_epoch: Ending epoch.
        focus_tile: Optional tile ID to track test loss specifically.
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
    criterion = nn.MSELoss()

    # If resuming, load previous checkpoint
    if start_epoch != 0:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'{start_epoch-1}_model.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=TORCH_DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Training loop
    for epoch in range(start_epoch, end_epoch):
        print(f"\nEpoch {epoch} starting...")
        train_loss = train_model(model, train_dataloader, optimizer, criterion)
        test_loss, bilinear_test_loss, focus_tile_loss, focus_tile_bilinear_loss = test_model(model, test_dataloader, criterion, focus_tile)

        print(f'Epoch {epoch}: Train loss = {train_loss}, Test loss = {test_loss}, Bilinear test loss = {bilinear_test_loss}')

        if focus_tile is not None:
            if focus_tile_loss is not None:
                if focus_tile_bilinear_loss is not None:
                    print(f"Epoch {epoch}: Test loss for tile {focus_tile} = {focus_tile_loss}, Bilinear test loss for tile {focus_tile} = {focus_tile_bilinear_loss}")
                else:
                    print(f"Epoch {epoch}: Test loss for tile {focus_tile} = {focus_tile_loss}")
            else:
                print(f"Epoch {epoch}: No samples found for tile {focus_tile} in the test set.")

        # Save checkpoint after each epoch
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'bilinear_test_loss': bilinear_test_loss
        }, os.path.join(CHECKPOINTS_DIR, f'{epoch}_model.pt'))
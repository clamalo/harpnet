# File: ./fine_tuning.py
"""
Fine-tuning script for multiple tiles.

This script fine-tunes the global model on each tile in a specified list of tiles, one by one.
All parameters are set within this file. Just run `python fine_tuning.py`.

To adjust parameters:
- Modify TILES_TO_FINE_TUNE to the list of tiles you want to fine-tune.
- Modify FINE_TUNE_EPOCHS and INITIAL_CHECKPOINT as needed.

The script:
1. Loads the global/best checkpoint.
2. For each tile in TILES_TO_FINE_TUNE:
   - Loads dataloaders focused on that tile.
   - Fine-tunes the model for FINE_TUNE_EPOCHS.
   - Saves checkpoints after each epoch.
   - Runs ensemble logic to find the best checkpoint for that tile.
   - Moves on to the next tile in the list.
"""

import os
import random
import numpy as np
import torch
from pathlib import Path

from src.constants import (CHECKPOINTS_DIR, TORCH_DEVICE, RANDOM_SEED, MODEL_NAME)
import importlib
from src.generate_dataloaders import generate_dataloaders
from src.ensemble import run_ensemble_on_directory

model_module = importlib.import_module(f"src.models.{MODEL_NAME}")
ModelClass = model_module.Model

# User-adjustable parameters
TILES_TO_FINE_TUNE = [4]  # List of tile indices to fine-tune
FINE_TUNE_EPOCHS = 5
INITIAL_CHECKPOINT = CHECKPOINTS_DIR / 'best' / 'best_model.pt'

def train_one_epoch(model: torch.nn.Module, 
                    train_dataloader, 
                    optimizer: torch.optim.Optimizer, 
                    criterion: torch.nn.Module, 
                    device: str) -> float:
    model.train()
    train_losses = []
    for batch in train_dataloader:
        inputs, elev_data, targets, times, tile_ids = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    return float(np.mean(train_losses)) if train_losses else float('inf')

def test_model(model: torch.nn.Module,
               test_dataloader, 
               criterion: torch.nn.Module, 
               device: str) -> (float, float):
    model.eval()
    test_losses = []
    bilinear_test_losses = []
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, elev_data, targets, times, tile_ids = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_losses.append(loss.item())

            # Bilinear baseline
            cropped_inputs = inputs[:,0:1,1:-1,1:-1]
            interpolated_inputs = torch.nn.functional.interpolate(cropped_inputs, size=(64,64), mode='bilinear')
            bilinear_loss = criterion(interpolated_inputs, targets)
            bilinear_test_losses.append(bilinear_loss.item())

    mean_test_loss = float(np.mean(test_losses)) if test_losses else float('inf')
    mean_bilinear_test_loss = float(np.mean(bilinear_test_losses)) if bilinear_test_losses else float('inf')
    return mean_test_loss, mean_bilinear_test_loss

def fine_tune_single_tile(tile: int,
                          fine_tune_epochs: int,
                          initial_checkpoint: Path):
    device = TORCH_DEVICE
    tile_ckpt_dir = CHECKPOINTS_DIR / str(tile)
    os.makedirs(tile_ckpt_dir, exist_ok=True)

    # Load dataloaders with focus on this tile
    train_dataloader, test_dataloader = generate_dataloaders(focus_tile=tile)

    model = ModelClass().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    if not initial_checkpoint.exists():
        raise FileNotFoundError(f"Initial checkpoint {initial_checkpoint} not found.")
    checkpoint = torch.load(initial_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    print(f"Starting fine-tuning for tile {tile} from checkpoint {initial_checkpoint} for {fine_tune_epochs} epochs.")
    for epoch in range(fine_tune_epochs):
        print(f"Tile {tile}, Epoch {epoch}...")
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
        test_loss, bilinear_test_loss = test_model(model, test_dataloader, criterion, device)
        print(f'Epoch {epoch}: Train loss = {train_loss}, Test loss = {test_loss}, Bilinear test = {bilinear_test_loss}')

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'bilinear_test_loss': bilinear_test_loss
        }, tile_ckpt_dir / f'{epoch}_model.pt')

    best_output_path = CHECKPOINTS_DIR / 'best' / f"{tile}_best.pt"
    run_ensemble_on_directory(str(tile_ckpt_dir), test_dataloader, device, str(best_output_path))
    print(f"Best fine-tuned model for tile {tile} saved at {best_output_path}")

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Fine-tune each tile in the list one by one
    for tile in TILES_TO_FINE_TUNE:
        fine_tune_single_tile(
            tile=tile,
            fine_tune_epochs=FINE_TUNE_EPOCHS,
            initial_checkpoint=INITIAL_CHECKPOINT
        )
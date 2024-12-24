"""
Provides a fine-tuning script for specific tiles, extending from a global best model.
Employs the hybrid loss and training functions defined in train_test.py,
and uses ensemble logic to find the best tile-specific model.
"""

import os
import random
import numpy as np
import torch
import logging
from pathlib import Path

from src.constants import (CHECKPOINTS_DIR, TORCH_DEVICE, RANDOM_SEED, MODEL_NAME, DETERMINISTIC)
import importlib
from src.generate_dataloaders import generate_dataloaders
from src.ensemble import ensemble_checkpoints
from src.train_test import train_one_epoch, test_model, get_criterion

# --- DYNAMIC MODEL IMPORT ---
model_module = importlib.import_module(f"src.models.{MODEL_NAME}")
ModelClass = model_module.Model

# --- USER-ADJUSTABLE PARAMETERS ---
TILES_TO_FINE_TUNE = [10]
FINE_TUNE_EPOCHS = 5
INITIAL_CHECKPOINT = CHECKPOINTS_DIR / 'best' / 'best_model.pt'

def fine_tune_single_tile(tile: int,
                          fine_tune_epochs: int,
                          initial_checkpoint: Path):
    """
    Fine-tunes the global best model on a specific tile for the specified number of epochs,
    then runs an ensemble over the newly created checkpoints to find the tile's best model.
    """
    device = TORCH_DEVICE
    tile_ckpt_dir = CHECKPOINTS_DIR / str(tile)
    os.makedirs(tile_ckpt_dir, exist_ok=True)

    train_dataloader, test_dataloader = generate_dataloaders(focus_tile=tile)

    model = ModelClass().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = get_criterion()

    # --- LOAD GLOBAL BEST MODEL AS STARTING POINT ---
    if not initial_checkpoint.exists():
        raise FileNotFoundError(f"Initial checkpoint {initial_checkpoint} not found.")
    checkpoint = torch.load(initial_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    logging.info(f"Starting fine-tuning for tile {tile} from checkpoint {initial_checkpoint} for {fine_tune_epochs} epochs.")
    for epoch in range(fine_tune_epochs):
        logging.info(f"Tile {tile}, Epoch {epoch}...")
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion)
        metrics = test_model(model, test_dataloader, criterion, focus_tile=tile)

        mean_test_loss = metrics["mean_test_loss"]
        mean_bilinear_loss = metrics["mean_bilinear_loss"]
        focus_tile_test_loss = metrics["focus_tile_test_loss"]
        focus_tile_bilinear_loss = metrics["focus_tile_bilinear_loss"]
        unnorm_focus_tile_mse = metrics["unnorm_focus_tile_mse"]
        unnorm_focus_tile_bilinear_mse = metrics["unnorm_focus_tile_bilinear_mse"]
        unnorm_focus_tile_mae = metrics["unnorm_focus_tile_mae"]
        unnorm_focus_tile_bilinear_mae = metrics["unnorm_focus_tile_bilinear_mae"]
        unnorm_focus_tile_corr = metrics["unnorm_focus_tile_corr"]
        unnorm_focus_tile_bilinear_corr = metrics["unnorm_focus_tile_bilinear_corr"]

        logging.info(f'Epoch {epoch}: Train loss (normalized Hybrid) = {train_loss:.6f}')
        logging.info(f'  Test (normalized Hybrid): {mean_test_loss:.6f}, Bilinear: {mean_bilinear_loss:.6f}')
        if focus_tile_test_loss is not None:
            logging.info(f'  Focus Tile {tile} (normalized Hybrid): {focus_tile_test_loss:.6f}, Bilinear: {focus_tile_bilinear_loss:.6f}')
            logging.info(f'  Focus Tile {tile} Unnorm MSE: {unnorm_focus_tile_mse:.6f}, Bilinear: {unnorm_focus_tile_bilinear_mse:.6f}')
            logging.info(f'  Focus Tile {tile} Unnorm MAE: {unnorm_focus_tile_mae:.6f}, Bilinear: {unnorm_focus_tile_bilinear_mae:.6f}')
            logging.info(f'  Focus Tile {tile} Unnorm Corr: {unnorm_focus_tile_corr:.4f}, Bilinear: {unnorm_focus_tile_bilinear_corr:.4f}')

        # --- SAVE CHECKPOINT ---
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': mean_test_loss,
            'bilinear_test_loss': mean_bilinear_loss
        }, tile_ckpt_dir / f'{epoch}_model.pt')

    # --- RUN ENSEMBLE ON TILE-SPECIFIC CHECKPOINTS ---
    best_output_path = CHECKPOINTS_DIR / 'best' / f"{tile}_best.pt"
    ensemble_checkpoints(
        test_dataloader=test_dataloader,
        device=device,
        directory_path=str(tile_ckpt_dir),
        output_path=str(best_output_path),
        max_ensemble_size=None
    )
    logging.info(f"Best fine-tuned model for tile {tile} saved at {best_output_path}")

if __name__ == "__main__":
    # --- SEED & DETERMINISTIC BEHAVIOR ---
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    # --- FINE-TUNE EACH TILE IN TILES_TO_FINE_TUNE ---
    for tile in TILES_TO_FINE_TUNE:
        fine_tune_single_tile(
            tile=tile,
            fine_tune_epochs=FINE_TUNE_EPOCHS,
            initial_checkpoint=INITIAL_CHECKPOINT
        )
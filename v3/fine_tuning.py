"""
Fine-tuning now relies on functions from train_test.py for training and evaluation.
This removes redundancy in training and testing code.
"""

import os
import random
import numpy as np
import torch
import logging
from pathlib import Path

from src.constants import (CHECKPOINTS_DIR, TORCH_DEVICE, RANDOM_SEED, MODEL_NAME)
import importlib
from src.generate_dataloaders import generate_dataloaders
from src.ensemble import run_ensemble_on_directory
from src.train_test import train_one_epoch, test_model

model_module = importlib.import_module(f"src.models.{MODEL_NAME}")
ModelClass = model_module.Model

# User-adjustable parameters
TILES_TO_FINE_TUNE = [10]  # List of tile indices to fine-tune
FINE_TUNE_EPOCHS = 5
INITIAL_CHECKPOINT = CHECKPOINTS_DIR / 'best' / 'best_model.pt'

def fine_tune_single_tile(tile: int,
                          fine_tune_epochs: int,
                          initial_checkpoint: Path):
    device = TORCH_DEVICE
    tile_ckpt_dir = CHECKPOINTS_DIR / str(tile)
    os.makedirs(tile_ckpt_dir, exist_ok=True)

    train_dataloader, test_dataloader = generate_dataloaders(focus_tile=tile)

    model = ModelClass().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

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

        logging.info(f'Epoch {epoch}: Train loss (normalized MSE) = {train_loss:.6f}')
        logging.info(f'  Test (normalized MSE): {mean_test_loss:.6f}, Bilinear: {mean_bilinear_loss:.6f}')
        if focus_tile_test_loss is not None:
            logging.info(f'  Focus Tile {tile} (normalized MSE): {focus_tile_test_loss:.6f}, Bilinear: {focus_tile_bilinear_loss:.6f}')
            logging.info(f'  Focus Tile {tile} Unnorm MSE: {unnorm_focus_tile_mse:.6f}, Bilinear: {unnorm_focus_tile_bilinear_mse:.6f}')
            logging.info(f'  Focus Tile {tile} Unnorm MAE: {unnorm_focus_tile_mae:.6f}, Bilinear: {unnorm_focus_tile_bilinear_mae:.6f}')
            logging.info(f'  Focus Tile {tile} Unnorm Corr: {unnorm_focus_tile_corr:.4f}, Bilinear: {unnorm_focus_tile_bilinear_corr:.4f}')

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': mean_test_loss,
            'bilinear_test_loss': mean_bilinear_loss
        }, tile_ckpt_dir / f'{epoch}_model.pt')

    best_output_path = CHECKPOINTS_DIR / 'best' / f"{tile}_best.pt"
    run_ensemble_on_directory(str(tile_ckpt_dir), test_dataloader, device, str(best_output_path))
    logging.info(f"Best fine-tuned model for tile {tile} saved at {best_output_path}")


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
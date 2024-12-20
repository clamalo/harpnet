# File: ./fine_tuning.py
"""
Fine-tuning script for individual tiles.
Trains from the best global model weights (found by ensemble) and fine-tunes on a single tile's data.
After fine-tuning, uses the run_ensemble_on_directory function from ensemble.py to find the best fine-tuned model.

This file is now placed at the same level as train.py.
"""

import os
import random
import numpy as np
import torch
from typing import List
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from src.constants import (CHECKPOINTS_DIR, PROCESSED_DIR, TORCH_DEVICE, RANDOM_SEED, MODEL_NAME)
import importlib
from src.generate_dataloaders import generate_dataloaders  # only for structural reference, we don't actually use this here
from src.ensemble import run_ensemble_on_directory

# Dynamically import the chosen model
model_module = importlib.import_module(f"src.models.{MODEL_NAME}")
ModelClass = model_module.Model

class SingleTileDataset(Dataset):
    def __init__(self, inputs: np.ndarray,
                 targets: np.ndarray,
                 times: np.ndarray,
                 tile_ids: np.ndarray,
                 tile_elev: np.ndarray,
                 tile_id_to_index: dict,
                 focus_tile: int):
        self.focus_tile = focus_tile
        mask = (tile_ids == self.focus_tile)
        self.inputs = inputs[mask]
        self.targets = targets[mask]
        self.times = times[mask]
        self.tile_ids = tile_ids[mask]
        self.tile_idx = tile_id_to_index[self.focus_tile]
        self.tile_elev = tile_elev[self.tile_idx]  # (1,Hf,Wf)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        input_data = torch.from_numpy(self.inputs[idx])   # (C,H,W)
        target_data = torch.from_numpy(self.targets[idx]) # (1,H,W)
        time_data = self.times[idx]
        tile_data = self.tile_ids[idx]
        elev_data = torch.from_numpy(self.tile_elev)      # (1,Hf,Wf)
        return input_data, elev_data, target_data, time_data, tile_data

def generate_single_tile_dataloaders(tile: int, train_test_ratio: float):
    train_input = np.load(PROCESSED_DIR / "combined_train_input.npy", mmap_mode='r')
    train_target = np.load(PROCESSED_DIR / "combined_train_target.npy", mmap_mode='r')
    train_times = np.load(PROCESSED_DIR / "combined_train_times.npy", mmap_mode='r')
    train_tile_ids = np.load(PROCESSED_DIR / "combined_train_tile_ids.npy", mmap_mode='r')

    test_input = np.load(PROCESSED_DIR / "combined_test_input.npy", mmap_mode='r')
    test_target = np.load(PROCESSED_DIR / "combined_test_target.npy", mmap_mode='r')
    test_times = np.load(PROCESSED_DIR / "combined_test_times.npy", mmap_mode='r')
    test_tile_ids = np.load(PROCESSED_DIR / "combined_test_tile_ids.npy", mmap_mode='r')

    tile_elev = np.load(PROCESSED_DIR / "combined_tile_elev.npy")
    unique_tile_ids = np.unique(np.concatenate([train_tile_ids, test_tile_ids]))
    tile_id_to_index = {t: i for i, t in enumerate(unique_tile_ids)}

    train_dataset = SingleTileDataset(train_input, train_target, train_times, train_tile_ids, tile_elev, tile_id_to_index, tile)
    test_dataset = SingleTileDataset(test_input, test_target, test_times, test_tile_ids, tile_elev, tile_id_to_index, tile)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(RANDOM_SEED)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=loader_generator, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    return train_dataloader, test_dataloader

def train_one_epoch(model: torch.nn.Module, 
                    train_dataloader, 
                    optimizer: torch.optim.Optimizer, 
                    criterion: torch.nn.Module, 
                    device: str) -> float:
    model.train()
    train_losses = []
    for batch in train_dataloader:
        inputs, elev_data, targets, times, tile_ids = batch
        inputs = torch.nn.functional.interpolate(inputs, size=(64,64), mode='nearest')
        elev_data = torch.nn.functional.interpolate(elev_data, size=(64,64), mode='nearest')
        inputs = torch.cat([inputs, elev_data], dim=1)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    return np.mean(train_losses) if train_losses else float('inf')

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
            inputs = torch.nn.functional.interpolate(inputs, size=(64,64), mode='nearest')
            elev_data = torch.nn.functional.interpolate(elev_data, size=(64,64), mode='nearest')
            inputs = torch.cat([inputs, elev_data], dim=1)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_losses.append(loss.item())

            # Bilinear baseline
            cropped_inputs = inputs[:,0:1,1:-1,1:-1]
            interpolated_inputs = torch.nn.functional.interpolate(cropped_inputs, size=(64,64), mode='bilinear')
            bilinear_loss = criterion(interpolated_inputs, targets)
            bilinear_test_losses.append(bilinear_loss.item())

    mean_test_loss = np.mean(test_losses) if test_losses else float('inf')
    mean_bilinear_test_loss = np.mean(bilinear_test_losses) if bilinear_test_losses else float('inf')
    return mean_test_loss, mean_bilinear_test_loss

def fine_tune_tile(tile: int,
                   fine_tune_epochs: int=10,
                   initial_checkpoint: str = None):
    device = TORCH_DEVICE
    tile_ckpt_dir = CHECKPOINTS_DIR / str(tile)
    os.makedirs(tile_ckpt_dir, exist_ok=True)

    train_dataloader, test_dataloader = generate_single_tile_dataloaders(tile, train_test_ratio=0.2)

    model = ModelClass().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    # Load initial weights from the best ensemble model
    if initial_checkpoint is None:
        best_global_ckpt = CHECKPOINTS_DIR / 'best' / 'best_model.pt'
    else:
        best_global_ckpt = Path(initial_checkpoint)
    if not best_global_ckpt.exists():
        raise FileNotFoundError(f"Initial checkpoint {best_global_ckpt} not found.")
    checkpoint = torch.load(best_global_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Train for specified epochs
    for epoch in range(fine_tune_epochs):
        print(f"Tile {tile}, Epoch {epoch} starting...")
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
        test_loss, bilinear_test_loss = test_model(model, test_dataloader, criterion, device)
        print(f'Tile {tile}, Epoch {epoch}: Train loss = {train_loss}, Test loss = {test_loss}, Bilinear test = {bilinear_test_loss}')

        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'bilinear_test_loss': bilinear_test_loss
        }, tile_ckpt_dir / f'{epoch}_model.pt')

    # After finishing, run the ensemble logic on the tile directory
    best_output_path = CHECKPOINTS_DIR / 'best' / f"{tile}_best.pt"
    # Reuse run_ensemble_on_directory from ensemble.py
    run_ensemble_on_directory(str(tile_ckpt_dir), test_dataloader, device, str(best_output_path))
    print(f"Best fine-tuned model for tile {tile} saved at {best_output_path}")

if __name__ == "__main__":
    # Example usage:
    fine_tune_tiles = [3, 7]
    fine_tune_epochs = 10
    initial_checkpoint = CHECKPOINTS_DIR / 'best' / 'best_model.pt'

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    for tile in fine_tune_tiles:
        fine_tune_tile(tile, fine_tune_epochs=fine_tune_epochs, initial_checkpoint=str(initial_checkpoint))
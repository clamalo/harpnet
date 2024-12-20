# File: /src/ensemble.py
"""
Implements model ensembling by averaging model weights from multiple checkpoints.
Evaluates the performance on a test set and saves the best ensemble model.

This file now includes the run_ensemble_on_directory function again to be used by fine_tuning.py.
"""

import os
import gc
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Tuple, Dict
from src.generate_dataloaders import generate_dataloaders
from src.constants import CHECKPOINTS_DIR, TORCH_DEVICE, MODEL_NAME
import importlib

# Dynamic import of model based on MODEL_NAME from constants
model_module = importlib.import_module(f"src.models.{MODEL_NAME}")
ModelClass = model_module.Model

def load_checkpoint_test_loss(checkpoint_path: str, device: str) -> Tuple[float, Dict[str, torch.Tensor], float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    bilinear_test_loss = checkpoint['bilinear_test_loss']

    # Extract test_loss robustly
    if 'test_loss' in checkpoint:
        test_loss = checkpoint['test_loss']
    elif 'test_losses' in checkpoint:
        test_loss = checkpoint['test_losses']
        if isinstance(test_loss, list):
            test_loss = sum(test_loss) / len(test_loss)
        elif isinstance(test_loss, dict):
            test_loss = test_loss.get('mse_loss', None)
            if test_loss is None:
                raise KeyError(f"'mse_loss' not found in 'test_losses' for checkpoint {checkpoint_path}.")
        else:
            raise TypeError(f"Unexpected type for 'test_losses' in checkpoint {checkpoint_path}: {type(test_loss)}")
    else:
        raise KeyError(f"'test_loss' key not found in checkpoint {checkpoint_path}.")

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    if test_loss is None:
        raise ValueError(f"Test loss could not be determined for checkpoint {checkpoint_path}.")

    return test_loss, state_dict, bilinear_test_loss

def initialize_cumulative_state_dict(state_dict: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    cumulative = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cumulative[key] = torch.zeros_like(value, dtype=torch.float32, device=device)
    return cumulative

def add_state_dict_to_cumulative(cumulative: Dict[str, torch.Tensor], new_state: Dict[str, torch.Tensor]) -> None:
    for key in cumulative.keys():
        if key in new_state and isinstance(new_state[key], torch.Tensor):
            cumulative[key] += new_state[key].float()

def evaluate_ensemble(model: ModelClass, test_dataloader, device: str) -> float:
    loss_fn = torch.nn.MSELoss()
    total_loss = 0.0
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, elev_data, targets, times, tile_ids = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

    mean_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return mean_loss

def ensemble(tiles: List[int], 
             start_month: Tuple[int,int], 
             end_month: Tuple[int,int], 
             train_test_ratio: float, 
             max_ensemble_size: Optional[int] = None) -> None:
    device = TORCH_DEVICE
    print(f"Using device: {device}")

    print("Generating data loaders for all tiles combined...")
    train_dataloader, test_dataloader = generate_dataloaders()
    print(f"Number of test batches: {len(test_dataloader)}")

    print("Initializing the model...")
    model = ModelClass().to(device)

    checkpoint_files = [
        f for f in os.listdir(CHECKPOINTS_DIR)
        if os.path.isfile(os.path.join(CHECKPOINTS_DIR, f)) and f.endswith('_model.pt')
    ]

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {CHECKPOINTS_DIR}")

    checkpoints = []
    print(f"Found {len(checkpoint_files)} checkpoint file(s).")

    # Load and record test losses
    for file_name in checkpoint_files:
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, file_name)
        try:
            test_loss, _, bilinear_test_loss = load_checkpoint_test_loss(checkpoint_path, device)
            checkpoints.append({
                'file_name': file_name,
                'test_loss': test_loss,
                'checkpoint_path': checkpoint_path,
                'bilinear_test_loss': bilinear_test_loss
            })
            print(f"Loaded checkpoint '{file_name}' with test_loss: {test_loss}")
        except (KeyError, TypeError, ValueError) as e:
            print(f"Skipping checkpoint '{file_name}': {e}")

    if not checkpoints:
        raise ValueError("No valid checkpoints with test_loss found.")

    sorted_checkpoints = sorted(checkpoints, key=lambda x: x['test_loss'])
    print("\nCheckpoints sorted by test_loss (ascending):")
    for ckpt in sorted_checkpoints:
        print(f"  {ckpt['file_name']}: Test Loss = {ckpt['test_loss']}")

    total_models = len(sorted_checkpoints)
    print(f"\nTotal valid checkpoints to consider: {total_models}")

    if max_ensemble_size is not None:
        max_ensemble_size = min(max_ensemble_size, total_models)
        print(f"Maximum ensemble size set to: {max_ensemble_size}")
    else:
        max_ensemble_size = total_models
        print(f"No maximum ensemble size specified. Using all {max_ensemble_size} available checkpoints.")

    # Start averaging from the best checkpoint
    first_checkpoint = sorted_checkpoints[0]
    test_loss, state_dict, bilinear_test_loss = load_checkpoint_test_loss(first_checkpoint['checkpoint_path'], device)
    cumulative_state_dict = initialize_cumulative_state_dict(state_dict, device)
    add_state_dict_to_cumulative(cumulative_state_dict, state_dict)

    # Evaluate initial best model alone
    print(f"\nEvaluating ensemble with 1 model:")
    model.load_state_dict(state_dict)
    del state_dict
    gc.collect()
    mean_loss = evaluate_ensemble(model, test_dataloader, device)
    print(f"Ensemble with 1 model: Mean Loss = {mean_loss:.6f}")

    best_mean_loss = mean_loss
    best_num_models = 1
    best_ensemble_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    best_bilinear_test_loss = bilinear_test_loss

    # Incrementally add models to the ensemble
    for N in range(2, max_ensemble_size + 1):
        print(f"\nEvaluating ensemble with {N} model(s):")
        next_checkpoint = sorted_checkpoints[N-1]
        try:
            test_loss_n, state_dict_n, bilinear_test_loss_n = load_checkpoint_test_loss(next_checkpoint['checkpoint_path'], device)
            add_state_dict_to_cumulative(cumulative_state_dict, state_dict_n)
        except Exception as e:
            print(f"Failed to load state_dict from '{next_checkpoint['file_name']}': {e}")
            continue

        averaged_state_dict = {key: (val / N) for key, val in cumulative_state_dict.items()}

        try:
            model.load_state_dict(averaged_state_dict, strict=False)
        except Exception as e:
            print(f"Failed to load averaged state_dict into the model for ensemble size {N}: {e}")
            del averaged_state_dict
            del state_dict_n
            gc.collect()
            continue

        mean_loss = evaluate_ensemble(model, test_dataloader, device)
        print(f"Ensemble with {N} model(s): Mean Loss = {mean_loss:.6f}")

        if mean_loss < best_mean_loss:
            best_mean_loss = mean_loss
            best_num_models = N
            best_ensemble_state_dict = {key: val.clone() for key, val in averaged_state_dict.items()}
            best_bilinear_test_loss = bilinear_test_loss_n

        del averaged_state_dict
        del state_dict_n
        gc.collect()

    # Save the best ensemble state if found
    if best_ensemble_state_dict is not None:
        model.load_state_dict(best_ensemble_state_dict, strict=False)
        print(f"\nOptimal ensemble size: {best_num_models} model(s) with Mean Loss = {best_mean_loss:.6f}")

        best_dir = os.path.join(CHECKPOINTS_DIR, 'best')
        os.makedirs(best_dir, exist_ok=True)

        best_model_path = os.path.join(best_dir, f"best_model.pt")
        try:
            torch.save({
                'model_state_dict': best_ensemble_state_dict,
                'test_loss': best_mean_loss,
                'bilinear_test_loss': best_bilinear_test_loss
            }, best_model_path)
            print(f"Best ensemble model saved to: {best_model_path}")
        except Exception as e:
            print(f"Failed to save the best ensemble model: {e}")
    else:
        print("\nNo valid ensemble found.")


def run_ensemble_on_directory(directory_path: str, test_dataloader, device: str, output_path: str) -> str:
    """
    Runs the ensemble logic on all checkpoints found in the specified directory.
    Sorts them by test_loss, tries ensemble averaging from 1 to N models, and saves the best.

    Args:
        directory_path: The directory containing checkpoint files.
        test_dataloader: Dataloader for testing.
        device: Torch device.
        output_path: Path to save the best ensemble model.

    Returns:
        The path to the saved best ensemble model.
    """
    model = ModelClass().to(device)

    checkpoint_files = [
        f for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f)) and f.endswith('_model.pt')
    ]

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {directory_path}")

    checkpoints = []
    for file_name in checkpoint_files:
        checkpoint_path = os.path.join(directory_path, file_name)
        try:
            test_loss, state_dict, bilinear_test_loss = load_checkpoint_test_loss(checkpoint_path, device)
            checkpoints.append({
                'file_name': file_name,
                'test_loss': test_loss,
                'checkpoint_path': checkpoint_path,
                'bilinear_test_loss': bilinear_test_loss,
                'state_dict': state_dict
            })
        except Exception as e:
            print(f"Skipping checkpoint '{file_name}': {e}")

    if not checkpoints:
        raise ValueError("No valid checkpoints with test_loss found in the specified directory.")

    sorted_checkpoints = sorted(checkpoints, key=lambda x: x['test_loss'])

    # Start from the best single checkpoint
    first = sorted_checkpoints[0]
    cumulative_state_dict = initialize_cumulative_state_dict(first['state_dict'], device)
    add_state_dict_to_cumulative(cumulative_state_dict, first['state_dict'])

    model.load_state_dict(first['state_dict'], strict=False)
    best_mean_loss = evaluate_ensemble(model, test_dataloader, device)
    best_num_models = 1
    best_ensemble_state_dict = {k: v.clone() for k,v in model.state_dict().items()}
    best_bilinear = first['bilinear_test_loss']

    # Try ensembles of increasing size
    for N in range(2, len(sorted_checkpoints)+1):
        next_ckpt = sorted_checkpoints[N-1]
        add_state_dict_to_cumulative(cumulative_state_dict, next_ckpt['state_dict'])
        averaged_state_dict = {k: (v/N) for k,v in cumulative_state_dict.items()}
        try:
            model.load_state_dict(averaged_state_dict, strict=False)
        except Exception as e:
            print(f"Failed to load averaged state_dict for N={N}: {e}")
            del averaged_state_dict
            gc.collect()
            continue

        mean_loss = evaluate_ensemble(model, test_dataloader, device)
        if mean_loss < best_mean_loss:
            best_mean_loss = mean_loss
            best_num_models = N
            best_ensemble_state_dict = {key: val.clone() for key,val in model.state_dict().items()}
            best_bilinear = next_ckpt['bilinear_test_loss']

        del averaged_state_dict
        gc.collect()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.save({
        'model_state_dict': best_ensemble_state_dict,
        'test_loss': best_mean_loss,
        'bilinear_test_loss': best_bilinear
    }, output_path)

    print(f"Best ensemble for {directory_path} found with {best_num_models} model(s), saved to {output_path}.")
    return output_path
"""
Implements ensemble logic by averaging model weights from multiple checkpoints.
Leverages train_test.py for evaluation using a unified loss/metric pipeline.
"""

import os
import gc
import torch
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict
import importlib
import torch.nn as nn

from src.constants import CHECKPOINTS_DIR, TORCH_DEVICE, MODEL_NAME
from src.train_test import test_model, get_criterion
from src.generate_dataloaders import generate_dataloaders

# Dynamically import model based on MODEL_NAME from constants
model_module = importlib.import_module(f"src.models.{MODEL_NAME}")
ModelClass = model_module.Model

def load_checkpoint_test_loss(checkpoint_path: str, device: str) -> Tuple[float, Dict[str, torch.Tensor], float]:
    """
    Loads checkpoint data and extracts the normalized test loss, bilinear test loss, and model state dict.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    bilinear_test_loss = checkpoint['bilinear_test_loss']

    if 'test_loss' in checkpoint:
        test_loss = checkpoint['test_loss']
    elif 'test_losses' in checkpoint:
        test_loss = checkpoint['test_losses']
        if isinstance(test_loss, list):
            test_loss = sum(test_loss) / len(test_loss)
        elif isinstance(test_loss, dict):
            test_loss = test_loss.get('mse_loss', None)
            if test_loss is None:
                raise KeyError(f"'mse_loss' not found in 'test_losses' in checkpoint {checkpoint_path}.")
        else:
            raise TypeError(f"Unexpected type for 'test_losses' in checkpoint {checkpoint_path}.")
    else:
        raise KeyError(f"'test_loss' key not found in checkpoint {checkpoint_path}.")

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # Some older checkpoints might have the model's state dict at the top level
        state_dict = checkpoint

    if test_loss is None:
        raise ValueError(f"Test loss could not be determined for checkpoint {checkpoint_path}.")

    return test_loss, state_dict, bilinear_test_loss

def initialize_cumulative_state_dict(state_dict: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """
    Creates a cumulative dict of zeros with the same shape as the provided state dict.
    """
    cumulative = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cumulative[key] = torch.zeros_like(value, dtype=torch.float32, device=device)
    return cumulative

def add_state_dict_to_cumulative(cumulative: Dict[str, torch.Tensor], new_state: Dict[str, torch.Tensor]) -> None:
    """
    Adds weights from new_state to the running total in cumulative.
    """
    for key in cumulative.keys():
        if key in new_state and isinstance(new_state[key], torch.Tensor):
            cumulative[key] += new_state[key].float()

def evaluate_model(model: ModelClass, test_dataloader, device: str) -> float:
    """
    Evaluates ensemble model using test_model from train_test.py and returns normalized loss.
    """
    criterion = get_criterion()
    metrics = test_model(model, test_dataloader, criterion, focus_tile=None)
    logging.info("Ensemble Evaluation:")
    logging.info(f"  Normalized MSE: {metrics['mean_test_loss']:.6f}")
    logging.info(f"  Unnorm MSE: {metrics['unnorm_test_mse']:.6f}, Unnorm MAE: {metrics['unnorm_test_mae']:.6f}, Unnorm Corr: {metrics['unnorm_test_corr']:.4f}")
    return metrics["mean_test_loss"]

def ensemble_checkpoints(
    test_dataloader,
    device: str,
    directory_path: str,
    output_path: Optional[str] = None,
    max_ensemble_size: Optional[int] = None
) -> str:
    """
    Performs an ensemble of checkpoints found in 'directory_path'. Averages their state dicts
    based on ascending order of normalized test loss, evaluating after each addition. The best 
    averaged state dict (lowest test loss) is saved to 'output_path' if provided, or 
    directory_path/best/best_model.pt by default.

    Returns:
        str: The file path to the best ensemble model.
    """
    # -------------------------------------------------------------------------
    # Gather checkpoint files from the specified directory
    # -------------------------------------------------------------------------
    checkpoint_files = [
        f for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f)) and f.endswith('_model.pt')
    ]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {directory_path}")

    # -------------------------------------------------------------------------
    # Load test losses for each checkpoint
    # -------------------------------------------------------------------------
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
            logging.info(f"Loaded checkpoint '{file_name}' with normalized test_loss: {test_loss}")
        except (KeyError, TypeError, ValueError) as e:
            logging.warning(f"Skipping checkpoint '{file_name}': {e}")

    if not checkpoints:
        raise ValueError("No valid checkpoints with test_loss found in the specified directory.")

    # -------------------------------------------------------------------------
    # Sort by ascending test_loss
    # -------------------------------------------------------------------------
    sorted_checkpoints = sorted(checkpoints, key=lambda x: x['test_loss'])
    logging.info("\nCheckpoints sorted by test_loss (ascending):")
    for ckpt in sorted_checkpoints:
        logging.info(f"  {ckpt['file_name']}: Normalized Test Loss = {ckpt['test_loss']}")

    total_models = len(sorted_checkpoints)
    if max_ensemble_size is not None:
        max_ensemble_size = min(max_ensemble_size, total_models)
        logging.info(f"Maximum ensemble size set to: {max_ensemble_size}")
    else:
        max_ensemble_size = total_models
        logging.info(f"No maximum ensemble size specified. Using all {max_ensemble_size} available checkpoints.")

    # -------------------------------------------------------------------------
    # Initialize
    # -------------------------------------------------------------------------
    model = ModelClass().to(device)
    first_ckpt = sorted_checkpoints[0]
    cumulative_state_dict = initialize_cumulative_state_dict(first_ckpt['state_dict'], device)
    add_state_dict_to_cumulative(cumulative_state_dict, first_ckpt['state_dict'])

    # Load and evaluate first checkpoint
    logging.info(f"\nEvaluating ensemble with 1 model (checkpoint: {first_ckpt['file_name']}):")
    model.load_state_dict(first_ckpt['state_dict'], strict=False)
    del first_ckpt['state_dict']
    gc.collect()

    best_mean_loss = evaluate_model(model, test_dataloader, device)
    best_num_models = 1
    best_ensemble_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    best_bilinear_test_loss = sorted_checkpoints[0]['bilinear_test_loss']

    # -------------------------------------------------------------------------
    # Iteratively add checkpoints
    # -------------------------------------------------------------------------
    for N in range(2, max_ensemble_size + 1):
        next_ckpt = sorted_checkpoints[N - 1]
        logging.info(f"\nEvaluating ensemble with {N} model(s): {next_ckpt['file_name']}")
        try:
            add_state_dict_to_cumulative(cumulative_state_dict, next_ckpt['state_dict'])
            averaged_state_dict = {k: (v / N) for k, v in cumulative_state_dict.items()}
        except Exception as e:
            logging.warning(f"Skipping adding checkpoint '{next_ckpt['file_name']}': {e}")
            continue

        try:
            model.load_state_dict(averaged_state_dict, strict=False)
        except Exception as e:
            logging.warning(f"Failed to load averaged state_dict for ensemble size {N}: {e}")
            del averaged_state_dict
            del next_ckpt['state_dict']
            gc.collect()
            continue

        mean_loss = evaluate_model(model, test_dataloader, device)
        if mean_loss < best_mean_loss:
            best_mean_loss = mean_loss
            best_num_models = N
            best_ensemble_state_dict = {key: val.clone() for key, val in model.state_dict().items()}
            best_bilinear_test_loss = next_ckpt['bilinear_test_loss']

        del averaged_state_dict
        del next_ckpt['state_dict']
        gc.collect()

    # -------------------------------------------------------------------------
    # Save best ensemble
    # -------------------------------------------------------------------------
    if output_path is None:
        # Default to directory_path/best/best_model.pt
        best_dir = os.path.join(directory_path, 'best')
        os.makedirs(best_dir, exist_ok=True)
        output_path = os.path.join(best_dir, "best_model.pt")
    else:
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model.load_state_dict(best_ensemble_state_dict, strict=False)
    logging.info(f"\nOptimal ensemble size: {best_num_models} model(s) with Normalized MSE = {best_mean_loss:.6f}")

    try:
        torch.save({
            'model_state_dict': best_ensemble_state_dict,
            'test_loss': best_mean_loss,
            'bilinear_test_loss': best_bilinear_test_loss
        }, output_path)
        logging.info(f"Best ensemble model saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save the best ensemble model: {e}")

    return output_path
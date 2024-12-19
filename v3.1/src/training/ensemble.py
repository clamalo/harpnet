"""
Compute an ensemble of models by averaging their parameters.
"""

import os
import gc
import torch
import numpy as np
from typing import List, Optional, Tuple, Dict
from src.config import Config
from src.data.dataloaders import generate_dataloaders
from src.models.model import UNetWithAttention
from src.training.trainer import test_model
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def load_checkpoint_test_loss(checkpoint_path: str, device: str) -> Tuple[float, Dict[str, torch.Tensor], float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    bilinear_test_loss = checkpoint['bilinear_test_loss']
    test_loss = None
    if 'test_loss' in checkpoint:
        test_loss = checkpoint['test_loss']
    elif 'test_losses' in checkpoint:
        test_val = checkpoint['test_losses']
        if isinstance(test_val, list):
            test_loss = sum(test_val) / len(test_val)
        elif isinstance(test_val, dict):
            test_loss = test_val.get('mse_loss', None)
            if test_loss is None:
                raise KeyError(f"'mse_loss' not found in 'test_losses' for checkpoint {checkpoint_path}.")
        else:
            raise TypeError(f"Unexpected type for 'test_losses' in checkpoint {checkpoint_path}: {type(test_val)}")
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

def evaluate_ensemble(model: UNetWithAttention, test_dataloader, device: str) -> float:
    """
    Evaluate an ensemble model on a test dataloader.
    Returns mean loss.
    """
    loss_fn = torch.nn.MSELoss()
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_dataloader:
            inputs, elev_data, targets, times, tile_ids = batch
            inputs = torch.nn.functional.interpolate(inputs, size=(64, 64), mode='nearest')
            elev_data = torch.nn.functional.interpolate(elev_data, size=(64,64), mode='nearest')
            inputs = torch.cat([inputs, elev_data], dim=1)

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

    mean_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return mean_loss

def ensemble(
    tiles: List[int],
    start_month: Tuple[int,int],
    end_month: Tuple[int,int],
    train_test_ratio: float,
    max_ensemble_size: Optional[int] = None
) -> None:
    """
    Compute an ensemble of models by averaging their parameters.

    Args:
        tiles: list of tiles.
        start_month: (year, month) start.
        end_month: (year, month) end.
        train_test_ratio: Ratio for data split.
        max_ensemble_size: Optional max number of models to include in ensemble.
    """

    device = Config.TORCH_DEVICE
    logger.info(f"Using device: {device}")
    logger.info("Generating data loaders for all tiles combined...")
    train_dataloader, test_dataloader = generate_dataloaders(tiles, start_month, end_month, train_test_ratio)
    logger.info(f"Number of test batches: {len(test_dataloader)}")

    logger.info("Initializing the model...")
    model = UNetWithAttention().to(device)

    if not Config.CHECKPOINTS_DIR.exists():
        raise FileNotFoundError(f"Checkpoints directory does not exist: {Config.CHECKPOINTS_DIR}")

    checkpoint_files = [
        f for f in os.listdir(Config.CHECKPOINTS_DIR)
        if os.path.isfile(os.path.join(Config.CHECKPOINTS_DIR, f)) and f.endswith('_model.pt')
    ]

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {Config.CHECKPOINTS_DIR}")

    checkpoints = []
    logger.info(f"Found {len(checkpoint_files)} checkpoint file(s).")

    for file_name in checkpoint_files:
        checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, file_name)
        try:
            test_loss, _, bilinear_test_loss = load_checkpoint_test_loss(checkpoint_path, device)
            checkpoints.append({
                'file_name': file_name,
                'test_loss': test_loss,
                'checkpoint_path': checkpoint_path,
                'bilinear_test_loss': bilinear_test_loss
            })
            logger.info(f"Loaded checkpoint '{file_name}' with test_loss: {test_loss:.6f}")
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Skipping checkpoint '{file_name}': {e}")

    if not checkpoints:
        raise ValueError("No valid checkpoints with test_loss found.")

    sorted_checkpoints = sorted(checkpoints, key=lambda x: x['test_loss'])
    logger.info("Checkpoints sorted by test_loss (ascending):")
    for ckpt in sorted_checkpoints:
        logger.info(f"  {ckpt['file_name']}: Test Loss = {ckpt['test_loss']}")

    total_models = len(sorted_checkpoints)
    logger.info(f"Total valid checkpoints to consider: {total_models}")

    if max_ensemble_size is not None:
        max_ensemble_size = min(max_ensemble_size, total_models)
        logger.info(f"Maximum ensemble size set to: {max_ensemble_size}")
    else:
        max_ensemble_size = total_models
        logger.info(f"No maximum ensemble size specified. Using all {max_ensemble_size} models.")

    first_checkpoint = sorted_checkpoints[0]
    test_loss, state_dict, bilinear_test_loss = load_checkpoint_test_loss(first_checkpoint['checkpoint_path'], device)
    cumulative_state_dict = initialize_cumulative_state_dict(state_dict, device)
    add_state_dict_to_cumulative(cumulative_state_dict, state_dict)

    logger.info("Evaluating ensemble with 1 model:")
    model.load_state_dict(state_dict)
    del state_dict
    gc.collect()
    mean_loss = evaluate_ensemble(model, test_dataloader, device)
    logger.info(f"Ensemble with 1 model: Mean Loss = {mean_loss:.6f}")

    best_mean_loss = mean_loss
    best_num_models = 1
    best_ensemble_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    best_bilinear_test_loss = bilinear_test_loss

    for N in range(2, max_ensemble_size + 1):
        logger.info(f"Evaluating ensemble with {N} models:")
        next_checkpoint = sorted_checkpoints[N-1]
        try:
            test_loss_n, state_dict_n, bilinear_test_loss_n = load_checkpoint_test_loss(next_checkpoint['checkpoint_path'], device)
            add_state_dict_to_cumulative(cumulative_state_dict, state_dict_n)
        except Exception as e:
            logger.warning(f"Failed to load state_dict from '{next_checkpoint['file_name']}': {e}")
            continue

        averaged_state_dict = {key: (val / N) for key, val in cumulative_state_dict.items()}

        try:
            model.load_state_dict(averaged_state_dict, strict=False)
        except Exception as e:
            logger.warning(f"Failed to load averaged state_dict for ensemble size {N}: {e}")
            del averaged_state_dict
            del state_dict_n
            gc.collect()
            continue

        mean_loss = evaluate_ensemble(model, test_dataloader, device)
        logger.info(f"Ensemble with {N} models: Mean Loss = {mean_loss:.6f}")

        if mean_loss < best_mean_loss:
            best_mean_loss = mean_loss
            best_num_models = N
            best_ensemble_state_dict = {key: val.clone() for key, val in averaged_state_dict.items()}
            best_bilinear_test_loss = bilinear_test_loss_n

        del averaged_state_dict
        del state_dict_n
        gc.collect()

    if best_ensemble_state_dict is not None:
        model.load_state_dict(best_ensemble_state_dict, strict=False)
        logger.info(f"Optimal ensemble size: {best_num_models} models with Mean Loss = {best_mean_loss:.6f}")

        best_dir = os.path.join(Config.CHECKPOINTS_DIR, 'best')
        os.makedirs(best_dir, exist_ok=True)

        best_model_path = os.path.join(best_dir, f"best_model.pt")
        try:
            torch.save({
                'model_state_dict': best_ensemble_state_dict,
                'test_loss': best_mean_loss,
                'bilinear_test_loss': best_bilinear_test_loss
            }, best_model_path)
            logger.info(f"Best ensemble model saved to: {best_model_path}")
        except Exception as e:
            logger.error(f"Failed to save the best ensemble model: {e}")
    else:
        logger.warning("No valid ensemble found.")
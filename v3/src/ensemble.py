"""
Functions for creating and evaluating ensembles of models by averaging their parameters.
"""

import os
import gc
import torch
from tqdm import tqdm
from typing import List, Optional, Tuple, Dict, Any
from src.generate_dataloaders import generate_dataloaders
from src.model import UNetWithAttention
from src.constants import CHECKPOINTS_DIR, TORCH_DEVICE

def load_checkpoint_test_loss(checkpoint_path: str, device: str) -> Tuple[float, Dict[str, torch.Tensor], float]:
    """
    Load a checkpoint and retrieve the test loss and model state_dict.

    Returns:
        test_loss, state_dict, bilinear_test_loss
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    bilinear_test_loss = checkpoint['bilinear_test_loss']

    # Extract test_loss from checkpoint
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
        # In case old format directly saved model
        state_dict = checkpoint

    if test_loss is None:
        raise ValueError(f"Test loss could not be determined for checkpoint {checkpoint_path}.")

    return test_loss, state_dict, bilinear_test_loss

def initialize_cumulative_state_dict(state_dict: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """
    Initialize a cumulative state_dict to hold averaged parameters.
    """
    cumulative = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cumulative[key] = torch.zeros_like(value, dtype=torch.float32, device=device)
    return cumulative

def add_state_dict_to_cumulative(cumulative: Dict[str, torch.Tensor], new_state: Dict[str, torch.Tensor]) -> None:
    """
    Add a state_dict to the cumulative parameters.
    """
    for key in cumulative.keys():
        if key in new_state and isinstance(new_state[key], torch.Tensor):
            cumulative[key] += new_state[key].float()

def evaluate_ensemble(model: UNetWithAttention, test_dataloader, device: str) -> float:
    """
    Evaluate the ensemble model on the test dataloader and return the mean MSE loss.
    """
    loss_fn = torch.nn.MSELoss()
    total_loss = 0.0
    num_batches = 0

    model.eval()
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

def ensemble(tiles: List[int], 
             start_month: Tuple[int,int], 
             end_month: Tuple[int,int], 
             train_test_ratio: float, 
             max_ensemble_size: Optional[int] = None) -> None:
    """
    Compute an ensemble of models by averaging their parameters.
    Evaluate ensemble performance incrementally from 1 model up to max_ensemble_size.
    Save the best ensemble checkpoint to CHECKPOINTS_DIR/best/best_model.pt.
    """
    device = TORCH_DEVICE
    print(f"Using device: {device}")

    print("Generating data loaders for all tiles combined...")
    train_dataloader, test_dataloader = generate_dataloaders(tiles, start_month, end_month, train_test_ratio)
    print(f"Number of test batches: {len(test_dataloader)}")

    print("Initializing the model...")
    model = UNetWithAttention().to(device)

    # Find all checkpoint files
    checkpoint_files = [
        f for f in os.listdir(CHECKPOINTS_DIR)
        if os.path.isfile(os.path.join(CHECKPOINTS_DIR, f)) and f.endswith('_model.pt')
    ]

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {CHECKPOINTS_DIR}")

    checkpoints = []
    print(f"Found {len(checkpoint_files)} checkpoint file(s).")

    # Load checkpoints and test losses
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

    # Sort checkpoints by test_loss (best first)
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

    # Start building ensemble
    # Initialize with best single model
    first_checkpoint = sorted_checkpoints[0]
    test_loss, state_dict, bilinear_test_loss = load_checkpoint_test_loss(first_checkpoint['checkpoint_path'], device)
    cumulative_state_dict = initialize_cumulative_state_dict(state_dict, device)
    add_state_dict_to_cumulative(cumulative_state_dict, state_dict)

    # Evaluate ensemble with 1 model
    print(f"\nEvaluating ensemble with 1 model:")
    model.load_state_dict(state_dict)
    del state_dict
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    mean_loss = evaluate_ensemble(model, test_dataloader, device)
    print(f"Ensemble with 1 model: Mean Loss = {mean_loss:.6f}")

    best_mean_loss = mean_loss
    best_num_models = 1
    best_ensemble_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    best_bilinear_test_loss = bilinear_test_loss

    # Evaluate ensemble sizes from 2 up to max_ensemble_size
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
            if device == 'cuda':
                torch.cuda.empty_cache()
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
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Save best ensemble
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
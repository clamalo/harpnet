import os
import torch
from tqdm import tqdm
import gc
from src.generate_dataloaders import generate_dataloaders
from src.model import UNetWithAttention
from src.constants import raw_dir, processed_dir, checkpoints_dir, figures_dir, torch_device

def load_checkpoint_test_loss(checkpoint_path, device):
    """
    Loads a checkpoint and extracts the test loss along with the state dictionary.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        device (str): Device to map the checkpoint.

    Returns:
        tuple:
            float: Extracted test loss.
            dict: Extracted state dictionary.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle Checkpoint Structure
    if isinstance(checkpoint, dict):
        # Try to extract 'test_loss'; adjust the key name if necessary
        if 'test_loss' in checkpoint:
            test_loss = checkpoint['test_loss']
        elif 'test_losses' in checkpoint:
            # If 'test_losses' is a list or dict, extract appropriately
            test_loss = checkpoint['test_losses']
            # Example: if 'test_losses' is a list of losses per batch, take the mean
            if isinstance(test_loss, list):
                test_loss = sum(test_loss) / len(test_loss)
            elif isinstance(test_loss, dict):
                # If it's a dict with multiple metrics, adjust as needed
                test_loss = test_loss.get('mse_loss', None)
                if test_loss is None:
                    raise KeyError(f"'mse_loss' not found in 'test_losses' for checkpoint {checkpoint_path}.")
            else:
                raise TypeError(f"Unexpected type for 'test_losses' in checkpoint {checkpoint_path}: {type(test_loss)}")
        else:
            raise KeyError(f"'test_loss' or 'test_losses' key not found in checkpoint {checkpoint_path}.")

        # Extract the state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Assume the entire checkpoint is the state_dict
            state_dict = checkpoint
    else:
        # If checkpoint is not a dict, assume it's the state_dict
        test_loss = None  # Unable to extract test loss
        state_dict = checkpoint

    if test_loss is None:
        raise ValueError(f"Test loss could not be determined for checkpoint {checkpoint_path}.")

    return test_loss, state_dict

def initialize_cumulative_state_dict(state_dict, device):
    """
    Initializes a cumulative state dictionary with zeros based on the provided state_dict.

    Args:
        state_dict (dict): A state dictionary to base the cumulative dict on.
        device (str): Device to place the cumulative tensors.

    Returns:
        dict: A cumulative state dictionary initialized with zeros.
    """
    cumulative = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cumulative[key] = torch.zeros_like(value, dtype=torch.float32, device=device)
    return cumulative

def add_state_dict_to_cumulative(cumulative_state_dict, new_state_dict):
    """
    Adds a new state dictionary's parameters to the cumulative_state_dict.

    Args:
        cumulative_state_dict (dict): The cumulative sum of state dictionaries.
        new_state_dict (dict): The new state dictionary to add.

    Returns:
        None: The cumulative_state_dict is updated in place.
    """
    for key in cumulative_state_dict.keys():
        if key in new_state_dict and isinstance(new_state_dict[key], torch.Tensor):
            cumulative_state_dict[key] += new_state_dict[key].float()

def ensemble(tile, start_month, end_month, train_test_ratio, max_ensemble_size=None):
    """
    Automatically detects the optimal number of ensemble members based on test loss using a greedy selection approach,
    ensuring computational and memory efficiency by utilizing the MPS device exclusively.

    Args:
        tile (int): The specific tile identifier.
        start_month (tuple): Start month as (year, month), e.g., (1979, 10).
        end_month (tuple): End month as (year, month), e.g., (1980, 9).
        train_test_ratio (float): Ratio of training to testing data, e.g., 0.2 for 20% testing.
        max_ensemble_size (int, optional): The maximum number of top checkpoints to consider for ensembling.
                                           If None, all available checkpoints are considered.
    """
    # Construct Checkpoint Directory for the Tile
    checkpoint_tile_dir = os.path.join(checkpoints_dir, str(tile))
    if not os.path.isdir(checkpoint_tile_dir):
        raise FileNotFoundError(f"Checkpoint directory for tile {tile} does not exist: {checkpoint_tile_dir}")

    # Device Configuration
    device = torch_device
    print(f"Using device: {device}")

    # Generate Data Loaders
    print("Generating data loaders...")
    train_dataloader, test_dataloader = generate_dataloaders(tile, start_month, end_month, train_test_ratio)
    print(f"Number of test batches: {len(test_dataloader)}")

    # Initialize the Model
    print("Initializing the model...")
    model = UNetWithAttention(1, 1, output_shape=(64, 64)).to(device)
    model.eval()  # Set model to evaluation mode

    # Collect All Checkpoints and Their Test Losses
    checkpoint_files = [
        f for f in os.listdir(checkpoint_tile_dir)
        if os.path.isfile(os.path.join(checkpoint_tile_dir, f)) and f.endswith('_model.pt')
    ]

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_tile_dir}")

    checkpoints = []
    print(f"Found {len(checkpoint_files)} checkpoint file(s).")

    # Phase 1: Extract test_loss for each checkpoint without loading state_dict
    for file_name in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_tile_dir, file_name)
        try:
            test_loss, _ = load_checkpoint_test_loss(checkpoint_path, device)
            checkpoints.append({
                'file_name': file_name,
                'test_loss': test_loss,
                'checkpoint_path': checkpoint_path
            })
            print(f"Loaded checkpoint '{file_name}' with test_loss: {test_loss}")
        except (KeyError, TypeError, ValueError) as e:
            print(f"Skipping checkpoint '{file_name}': {e}")

    if not checkpoints:
        raise ValueError("No valid checkpoints with test_loss found.")

    # Sort checkpoints by test_loss in ascending order (lower loss is better)
    sorted_checkpoints = sorted(checkpoints, key=lambda x: x['test_loss'])
    print("\nCheckpoints sorted by test_loss (ascending):")
    for ckpt in sorted_checkpoints:
        print(f"  {ckpt['file_name']}: Test Loss = {ckpt['test_loss']}")

    total_models = len(sorted_checkpoints)
    print(f"\nTotal valid checkpoints to consider: {total_models}")

    # Determine the actual maximum ensemble size based on user input and available models
    if max_ensemble_size is not None:
        max_ensemble_size = min(max_ensemble_size, total_models)
        print(f"Maximum ensemble size set to: {max_ensemble_size}")
    else:
        max_ensemble_size = total_models
        print(f"No maximum ensemble size specified. Using all {max_ensemble_size} available checkpoints.")

    # Initialize variables to track the best ensemble
    best_num_models = 1
    best_mean_loss = float('inf')
    best_ensemble_state_dict = None
    bilinear_test_loss = None  # To store bilinear_test_loss

    # Phase 2: Greedy Selection
    # Start with the best single model
    first_checkpoint = sorted_checkpoints[0]
    try:
        test_loss, state_dict = load_checkpoint_test_loss(first_checkpoint['checkpoint_path'], device)
    except Exception as e:
        raise RuntimeError(f"Failed to load state_dict from '{first_checkpoint['file_name']}': {e}")

    # Initialize cumulative state dict
    cumulative_state_dict = initialize_cumulative_state_dict(state_dict, device)
    add_state_dict_to_cumulative(cumulative_state_dict, state_dict)

    # Start evaluating with the top 1 model
    print(f"\nEvaluating ensemble with 1 model:")
    model.load_state_dict(state_dict)
    del state_dict  # Free memory
    gc.collect()
    torch.cuda.empty_cache()

    # Define Loss Function
    loss_fn = torch.nn.MSELoss()

    # Initialize Loss Tracking
    total_loss = 0.0
    num_batches = 0

    # Disable Gradient Calculation for Evaluation
    with torch.no_grad():
        # Iterate Over Test Data with Progress Bar
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc='Evaluating Ensemble Size 1')):
            # Adjust the unpacking based on the number of elements in the batch
            if len(batch) == 2:
                inputs, targets = batch
            elif len(batch) == 3:
                inputs, targets, _ = batch  # Ignoring 'times'
            else:
                raise ValueError(f"Unexpected number of items in batch: {len(batch)}. Expected 2 or 3.")

            # Move Data to the Appropriate Device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward Pass
            outputs = model(inputs)

            # Compute Loss
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

    # Compute and Compare Mean Loss
    mean_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    print(f"Ensemble with 1 model: Mean Loss = {mean_loss:.6f}")

    if mean_loss < best_mean_loss:
        best_mean_loss = mean_loss
        best_num_models = 1
        best_ensemble_state_dict = model.state_dict()
        bilinear_test_loss = mean_loss  # Assuming bilinear_test_loss is the test_loss of the first model

    # Iterate over ensemble sizes from 2 to max_ensemble_size using Greedy Selection
    for N in range(2, max_ensemble_size + 1):
        print(f"\nEvaluating ensemble with {N} model(s):")

        # Select the next best model (greedy approach)
        next_checkpoint = sorted_checkpoints[N-1]
        try:
            test_loss, state_dict = load_checkpoint_test_loss(next_checkpoint['checkpoint_path'], device)
            add_state_dict_to_cumulative(cumulative_state_dict, state_dict)
        except Exception as e:
            print(f"Failed to load state_dict from '{next_checkpoint['file_name']}': {e}")
            continue

        # Compute the average state_dict
        averaged_state_dict = {}
        for key in cumulative_state_dict.keys():
            averaged_state_dict[key] = cumulative_state_dict[key] / N

        # Load the averaged weights into the model
        try:
            model.load_state_dict(averaged_state_dict, strict=False)
        except Exception as e:
            print(f"Failed to load averaged state_dict into the model for ensemble size {N}: {e}")
            del averaged_state_dict
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # Compute Loss
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc=f'Evaluating Ensemble Size {N}')):
                # Adjust the unpacking based on the number of elements in the batch
                if len(batch) == 2:
                    inputs, targets = batch
                elif len(batch) == 3:
                    inputs, targets, _ = batch  # Ignoring 'times'
                else:
                    raise ValueError(f"Unexpected number of items in batch: {len(batch)}. Expected 2 or 3.")

                # Move Data to the Appropriate Device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward Pass
                outputs = model(inputs)

                # Compute Loss
                loss = loss_fn(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        # Compute Mean Loss
        mean_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Ensemble with {N} model(s): Mean Loss = {mean_loss:.6f}")

        # Update best ensemble if current mean loss is better
        if mean_loss < best_mean_loss:
            best_mean_loss = mean_loss
            best_num_models = N
            best_ensemble_state_dict = {key: val.clone() for key, val in averaged_state_dict.items()}
            bilinear_test_loss = mean_loss  # Update bilinear_test_loss to current best

        # Clean up
        del averaged_state_dict
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

    # After evaluating all ensemble sizes, load the best ensemble into the model
    if best_ensemble_state_dict is not None:
        model.load_state_dict(best_ensemble_state_dict, strict=False)
        print(f"\nOptimal ensemble size: {best_num_models} model(s) with Mean Loss = {best_mean_loss:.6f}")

        # Save the best ensemble model
        best_dir = os.path.join(checkpoints_dir, 'best')
        os.makedirs(best_dir, exist_ok=True)  # Create directory if it doesn't exist

        best_model_path = os.path.join(best_dir, f"{tile}_model.pt")
        try:
            torch.save({
                'model_state_dict': best_ensemble_state_dict,
                'test_loss': best_mean_loss,
                'bilinear_test_loss': bilinear_test_loss
            }, best_model_path)
            print(f"Best ensemble model saved to: {best_model_path}")
        except Exception as e:
            print(f"Failed to save the best ensemble model to '{best_model_path}': {e}")
    else:
        print("\nNo valid ensemble found.")


if __name__ == '__main__':
    tile = 89
    start_month = (1979, 10)
    end_month = (2022, 9)
    train_test_ratio = 0.2
    max_ensemble_size = 5

    ensemble(tile, start_month, end_month, train_test_ratio, max_ensemble_size)
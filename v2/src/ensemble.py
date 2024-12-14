import os
import torch
from tqdm import tqdm
import gc
from src.generate_dataloaders import generate_dataloaders
from src.model import UNetWithAttention
from src.constants import CHECKPOINTS_DIR, TORCH_DEVICE, UNET_DEPTH

def load_checkpoint_test_loss(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    bilinear_test_loss = checkpoint['bilinear_test_loss']

    if isinstance(checkpoint, dict):
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
            raise KeyError(f"'test_loss' or 'test_losses' key not found in checkpoint {checkpoint_path}.")
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        test_loss = None
        state_dict = checkpoint

    if test_loss is None:
        raise ValueError(f"Test loss could not be determined for checkpoint {checkpoint_path}.")

    return test_loss, state_dict, bilinear_test_loss

def initialize_cumulative_state_dict(state_dict, device):
    cumulative = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cumulative[key] = torch.zeros_like(value, dtype=torch.float32, device=device)
    return cumulative

def add_state_dict_to_cumulative(cumulative_state_dict, new_state_dict):
    for key in cumulative_state_dict.keys():
        if key in new_state_dict and isinstance(new_state_dict[key], torch.Tensor):
            cumulative_state_dict[key] += new_state_dict[key].float()

def ensemble(tile, start_month, end_month, train_test_ratio, max_ensemble_size=None):
    checkpoint_tile_dir = os.path.join(CHECKPOINTS_DIR, str(tile))
    if not os.path.isdir(checkpoint_tile_dir):
        raise FileNotFoundError(f"Checkpoint directory for tile {tile} does not exist: {checkpoint_tile_dir}")

    device = TORCH_DEVICE
    print(f"Using device: {device}")

    print("Generating data loaders...")
    train_dataloader, test_dataloader = generate_dataloaders(tile, start_month, end_month, train_test_ratio)
    print(f"Number of test batches: {len(test_dataloader)}")

    print("Initializing the model...")
    model = UNetWithAttention().to(device)
    model.eval()

    checkpoint_files = [
        f for f in os.listdir(checkpoint_tile_dir)
        if os.path.isfile(os.path.join(checkpoint_tile_dir, f)) and f.endswith('_model.pt')
    ]

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_tile_dir}")

    checkpoints = []
    print(f"Found {len(checkpoint_files)} checkpoint file(s).")

    for file_name in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_tile_dir, file_name)
        try:
            test_loss, _, bilinear_test_loss = load_checkpoint_test_loss(checkpoint_path, device)
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

    best_num_models = 1
    best_mean_loss = float('inf')
    best_ensemble_state_dict = None
    bilinear_test_loss = None

    first_checkpoint = sorted_checkpoints[0]
    try:
        test_loss, state_dict, bilinear_test_loss = load_checkpoint_test_loss(first_checkpoint['checkpoint_path'], device)
    except Exception as e:
        raise RuntimeError(f"Failed to load state_dict from '{first_checkpoint['file_name']}': {e}")

    cumulative_state_dict = initialize_cumulative_state_dict(state_dict, device)
    add_state_dict_to_cumulative(cumulative_state_dict, state_dict)

    print(f"\nEvaluating ensemble with 1 model:")
    model.load_state_dict(state_dict)
    del state_dict
    gc.collect()
    torch.cuda.empty_cache()

    loss_fn = torch.nn.MSELoss()

    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc='Evaluating Ensemble Size 1')):
            if len(batch) == 2:
                inputs, targets = batch
            elif len(batch) == 3:
                inputs, targets, _ = batch
            else:
                raise ValueError(f"Unexpected number of items in batch: {len(batch)}. Expected 2 or 3.")

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

    mean_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    print(f"Ensemble with 1 model: Mean Loss = {mean_loss:.6f}")

    if mean_loss < best_mean_loss:
        best_mean_loss = mean_loss
        best_num_models = 1
        best_ensemble_state_dict = model.state_dict()
        bilinear_test_loss = bilinear_test_loss

    for N in range(2, max_ensemble_size + 1):
        print(f"\nEvaluating ensemble with {N} model(s):")
        next_checkpoint = sorted_checkpoints[N-1]
        try:
            test_loss, state_dict, bilinear_test_loss = load_checkpoint_test_loss(next_checkpoint['checkpoint_path'], device)
            add_state_dict_to_cumulative(cumulative_state_dict, state_dict)
        except Exception as e:
            print(f"Failed to load state_dict from '{next_checkpoint['file_name']}': {e}")
            continue

        averaged_state_dict = {}
        for key in cumulative_state_dict.keys():
            averaged_state_dict[key] = cumulative_state_dict[key] / N

        try:
            model.load_state_dict(averaged_state_dict, strict=False)
        except Exception as e:
            print(f"Failed to load averaged state_dict into the model for ensemble size {N}: {e}")
            del averaged_state_dict
            gc.collect()
            torch.cuda.empty_cache()
            continue

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc=f'Evaluating Ensemble Size {N}')):
                if len(batch) == 2:
                    inputs, targets = batch
                elif len(batch) == 3:
                    inputs, targets, _ = batch
                else:
                    raise ValueError(f"Unexpected number of items in batch: {len(batch)}. Expected 2 or 3.")

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        mean_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Ensemble with {N} model(s): Mean Loss = {mean_loss:.6f}")

        if mean_loss < best_mean_loss:
            best_mean_loss = mean_loss
            best_num_models = N
            best_ensemble_state_dict = {key: val.clone() for key, val in averaged_state_dict.items()}
            bilinear_test_loss = bilinear_test_loss

        del averaged_state_dict
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

    if best_ensemble_state_dict is not None:
        model.load_state_dict(best_ensemble_state_dict, strict=False)
        print(f"\nOptimal ensemble size: {best_num_models} model(s) with Mean Loss = {best_mean_loss:.6f}")

        best_dir = os.path.join(CHECKPOINTS_DIR, 'best')
        os.makedirs(best_dir, exist_ok=True)

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
import os
import math
import torch
import importlib
from pathlib import Path

from config import (
    CHECKPOINTS_DIR,
    MODEL_NAME,
    DEVICE,
    FINE_TUNE_EPOCHS
)
from generate_dataloaders import generate_dataloaders
from train_test import train, test
from ensemble import ensemble


def fine_tune_tiles(tile_ids, base_checkpoint_path, fine_tuning_epochs=FINE_TUNE_EPOCHS):
    """
    Fine-tune the model for each tile in tile_ids, using the best
    generalized model weights as initialization.

    Steps:
      1) Load the best generalized model from 'base_checkpoint_path'.
      2) For each tile_id:
         a) Create a subdirectory under CHECKPOINTS_DIR (e.g. "CHECKPOINTS_DIR / <tile_id>").
         b) Generate tile-specific train/test dataloaders by calling generate_dataloaders(tile_id=...).
         c) Train the model for 'fine_tuning_epochs'.
            - Save a checkpoint "fine_tune_{epoch}_model.pt" for each epoch
              with 'train_loss' and 'test_loss' in the state dict.
         d) Run ensemble on that subdirectory, which merges all these new checkpoints,
            selecting the best. We'll rename the resulting "best_model.pt" to "<tile_id>_best.pt"
            and put it in the "CHECKPOINTS_DIR / best" directory.

    Args:
        tile_ids (list[int]): List of tile IDs to fine-tune.
        base_checkpoint_path (str or Path): Path to the best generalized model checkpoint
                                            from the ensemble run.
        fine_tuning_epochs (int): How many epochs to train for each tile.
    """
    base_checkpoint_path = Path(base_checkpoint_path)
    if not base_checkpoint_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found at: {base_checkpoint_path}")

    # Decide on device
    if DEVICE.lower() == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif DEVICE.lower() == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            print("Warning: MPS requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    print(f"Fine-tuning on device: {device}")

    # Dynamically load the model
    model_module = importlib.import_module(MODEL_NAME)
    ModelClass = model_module.Model

    # Load the base checkpoint
    base_ckpt = torch.load(base_checkpoint_path, map_location=device)
    base_state_dict = base_ckpt.get('model_state_dict', None)
    if base_state_dict is None:
        raise ValueError(f"Could not load 'model_state_dict' from {base_checkpoint_path}")

    # Fine-tune for each tile
    for tile_id in tile_ids:
        print(f"\n--- Fine-tuning for tile {tile_id} ---")

        # Create subdirectory for this tile
        tile_subdir = CHECKPOINTS_DIR / str(tile_id)
        tile_subdir.mkdir(exist_ok=True, parents=True)

        # Prepare data
        train_loader, test_loader = generate_dataloaders(tile_id=tile_id)
        if len(train_loader) == 0:
            print(f"No training samples found for tile {tile_id}, skipping.")
            continue

        # Re-initialize model
        model = ModelClass().to(device)
        # Load the base generalized weights
        model.load_state_dict(base_state_dict, strict=True)

        # Define optimizer & loss
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        criterion = torch.nn.MSELoss()

        # Fine-tuning loop
        for epoch in range(fine_tuning_epochs):
            train_loss = train(train_loader, model, optimizer, criterion, device)
            test_loss = test(test_loader, model, criterion, device)

            # Save a checkpoint in tile_subdir
            checkpoint_path = tile_subdir / f"fine_tune_{epoch}_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss
            }, checkpoint_path)

            print(f"Tile {tile_id}, Epoch {epoch+1}/{fine_tuning_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # After we've created fine-tune checkpoints, run ensemble on them
        print(f"\nEnsembling checkpoints for tile {tile_id} ...")
        ensemble(tile_subdir)

        # The ensemble function saves its best model to tile_subdir / 'best' / 'best_model.pt'
        best_model_subdir = tile_subdir / "best" / "best_model.pt"
        if not best_model_subdir.exists():
            print(f"Warning: Could not find best_model.pt for tile {tile_id}.")
            continue

        # Rename/copy it to "CHECKPOINTS_DIR/best/<tile_id>_best.pt"
        best_dest_dir = CHECKPOINTS_DIR / "best"
        best_dest_dir.mkdir(exist_ok=True, parents=True)
        best_dest_path = best_dest_dir / f"{tile_id}_best.pt"

        # Copy (or move) the file
        torch.save(torch.load(best_model_subdir, map_location=device), best_dest_path)
        print(f"Tile {tile_id} best ensemble checkpoint saved to {best_dest_path}")

    print("\nFine-tuning complete for all specified tiles!")


if __name__ == "__main__":
    # Example usage
    fine_tune_tiles(
        tile_ids=[8, 15],
        base_checkpoint_path=CHECKPOINTS_DIR / 'best' / 'best_model.pt',
        fine_tuning_epochs=5
    )
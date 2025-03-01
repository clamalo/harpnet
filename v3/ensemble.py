from pathlib import Path
import math
import importlib
import torch

from config import (
    MODEL_NAME,
    DEVICE,
    MAX_ENSEMBLE_SIZE,
    CHECKPOINTS_DIR
)
from train_test import test
from generate_dataloaders import generate_dataloaders


def ensemble(directory_to_ensemble=None, focus_tile=None):
    """
    Find the optimal combination of model checkpoints by iteratively merging
    their weights and evaluating on the test set. By default (when both
    'directory_to_ensemble' and 'focus_tile' are None), this function searches
    in CHECKPOINTS_DIR and evaluates on the entire test set.

    If 'focus_tile' is specified, it searches in CHECKPOINTS_DIR / <focus_tile>
    and evaluates only on that tile (by passing tile_id=focus_tile to
    generate_dataloaders).

    Steps:
    1) Collect all checkpoint files in 'directory_to_ensemble'. Each file should be 
       of the form '{epoch}_model.pt', containing a 'test_loss' entry in the state dict.
    2) Sort checkpoints by ascending test_loss (the best single-model checkpoints first).
    3) For i in range(1, min(len(checkpoints), MAX_ENSEMBLE_SIZE) + 1):
         - Merge the top i checkpoints by averaging their weights at a parameter level.
         - Evaluate that merged model on the test set (for the entire domain or a single tile).
         - Keep track of whichever model (single or merged) yields the lowest test loss.
    4) Save the best merged weights to directory_to_ensemble / 'best' / 'best_model.pt'.

    The function reuses the 'test' routine from train_test.py. It calls
    generate_dataloaders() with tile_id=None if focus_tile is not provided,
    or tile_id=focus_tile if it is. The 'merge' operation is done at the
    parameter level by simply averaging each parameter tensor.

    Args:
        directory_to_ensemble (str or Path, optional):
            The directory to search for checkpoint files. If not provided,
            uses CHECKPOINTS_DIR. If 'focus_tile' is also provided, that
            takes precedence.
        focus_tile (int, optional):
            If provided, the ensemble function will search in
            CHECKPOINTS_DIR / <focus_tile> and evaluate only on that tile.
    """
    # ------------------------------------------------
    # 0) Determine ensemble directory and device
    # ------------------------------------------------
    if focus_tile is not None:
        directory_to_ensemble = CHECKPOINTS_DIR / str(focus_tile)
    elif directory_to_ensemble is None:
        directory_to_ensemble = CHECKPOINTS_DIR

    directory_to_ensemble = Path(directory_to_ensemble)

    print(f"Ensemble: using device: {DEVICE}")
    print(f"Ensemble directory: {directory_to_ensemble}")

    # ------------------------------------------------
    # 1) Gather & sort checkpoints by test_loss
    # ------------------------------------------------
    checkpoint_files = list(directory_to_ensemble.glob("*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {directory_to_ensemble}")

    ckpts_with_loss = []
    for ckpt_file in checkpoint_files:
        try:
            ckpt_data = torch.load(ckpt_file, map_location=DEVICE)
            test_loss = ckpt_data.get('test_loss', math.inf)
            ckpts_with_loss.append((ckpt_file, test_loss))
        except Exception as e:
            print(f"Skipping file {ckpt_file}: {e}")

    if not ckpts_with_loss:
        raise ValueError("No valid checkpoints found (couldn't read test_loss).")

    ckpts_with_loss.sort(key=lambda x: x[1])  # sort by ascending test_loss
    # Now ckpts_with_loss is a list of tuples: (Path, test_loss) sorted best -> worst

    print(f"Found {len(ckpts_with_loss)} checkpoints in {directory_to_ensemble}.")

    # We only consider up to the smaller of (MAX_ENSEMBLE_SIZE, number of checkpoints)
    top_ckpts = ckpts_with_loss[:min(len(ckpts_with_loss), MAX_ENSEMBLE_SIZE)]
    print(f"Will try ensembles up to size {len(top_ckpts)} based on test_loss ranking.")

    # ------------------------------------------------
    # 2) Create test dataloader
    # ------------------------------------------------
    if focus_tile is not None:
        _, test_loader = generate_dataloaders(tile_id=focus_tile)
        print(f"Evaluating ensemble performance ONLY on tile {focus_tile}.")
    else:
        _, test_loader = generate_dataloaders()
        print("Evaluating ensemble performance on the entire test set.")

    # Dynamically load the model
    model_module = importlib.import_module(MODEL_NAME)
    ModelClass = model_module.Model
    criterion = torch.nn.SmoothL1Loss()

    # Helper function to average multiple checkpoints
    def average_checkpoints(paths):
        """
        Load each checkpoint from 'paths', sum their weights, and then average them.
        Return a model instance with the merged (averaged) weights.
        """
        if not paths:
            raise ValueError("No checkpoint paths provided for averaging.")

        model_tmp = ModelClass().to(DEVICE)
        state_dict_sums = None
        n = len(paths)

        for p in paths:
            cp = torch.load(p, map_location=DEVICE)
            cp_st = cp['model_state_dict']
            if state_dict_sums is None:
                state_dict_sums = {key: val.clone() for key, val in cp_st.items()}
            else:
                for key, val in cp_st.items():
                    state_dict_sums[key] += val

        for key in state_dict_sums:
            state_dict_sums[key] /= float(n)

        model_tmp.load_state_dict(state_dict_sums)
        return model_tmp

    # ------------------------------------------------
    # 3) Incrementally merge top i checkpoints and track best
    # ------------------------------------------------
    best_loss = math.inf
    best_state_dict = None
    best_combo_size = 0

    for i in range(1, len(top_ckpts) + 1):
        paths_to_merge = [ckpt_file for (ckpt_file, _) in top_ckpts[:i]]
        merged_model = average_checkpoints(paths_to_merge)
        test_loss_val = test(test_loader, merged_model, criterion)
        print(f"Ensemble of top {i} model(s) => test_loss={test_loss_val:.6f}")

        if test_loss_val < best_loss:
            best_loss = test_loss_val
            best_combo_size = i
            best_state_dict = merged_model.state_dict()

    # ------------------------------------------------
    # 4) Save best merged model
    # ------------------------------------------------
    best_dir = directory_to_ensemble / "best"
    best_dir.mkdir(exist_ok=True)

    best_path = best_dir / "best_model.pt"
    torch.save({
        'ensemble_size': best_combo_size,
        'model_state_dict': best_state_dict,
        'test_loss': best_loss
    }, best_path)

    print(f"\nBest ensemble size: {best_combo_size}, Test Loss: {best_loss:.6f}")
    print(f"Saved best ensemble model to {best_path}")
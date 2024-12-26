from pathlib import Path
import math
import importlib
import torch

from config import (
    MODEL_NAME,
    DEVICE,
    MAX_ENSEMBLE_SIZE
)
from train_test import test
from generate_dataloaders import generate_dataloaders


def ensemble(directory_to_ensemble):
    """
    Find the optimal combination of model checkpoints by iteratively merging
    their weights and evaluating on the test set, searching inside
    'directory_to_ensemble' for checkpoint files.

    Steps:
    1) Collect all checkpoint files in 'directory_to_ensemble'. Each file should be of
       the form '{epoch}_model.pt', containing a 'test_loss' entry in the state dict.
    2) Sort checkpoints by ascending test_loss (the best single-model checkpoints first).
    3) For i in range(1, min(len(checkpoints), MAX_ENSEMBLE_SIZE) + 1):
         - Merge the top i checkpoints by averaging their weights at a parameter level.
         - Evaluate that merged model on the test set.
         - Keep track of whichever model (single or merged) yields the lowest test loss.
    4) Save the best merged weights to directory_to_ensemble / 'best' / 'best_model.pt'.

    The function re-uses the 'test' routine from train_test.py, and it automatically
    constructs the test dataloader by calling generate_dataloaders().

    Note: The merging is done incrementally from the top-1 up to the top-i, thus it
          always merges the top i distinct checkpoints in sorted order.
    """
    # -----------------------------
    # 0) Convert directory and set device
    # -----------------------------
    directory_to_ensemble = Path(directory_to_ensemble)

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

    print(f"Ensemble: using device: {device}")

    # -----------------------------
    # 1) Gather & sort checkpoints by test_loss
    # -----------------------------
    checkpoint_files = list(directory_to_ensemble.glob("*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {directory_to_ensemble}")

    ckpts_with_loss = []
    for ckpt_file in checkpoint_files:
        try:
            ckpt_data = torch.load(ckpt_file, map_location=device)
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

    # -----------------------------
    # 2) Create test dataloader
    # -----------------------------
    _, test_loader = generate_dataloaders()

    # We'll need a model class and loss function
    model_module = importlib.import_module(MODEL_NAME)
    ModelClass = model_module.Model
    criterion = torch.nn.MSELoss()

    # Helper function to average multiple checkpoints
    def average_checkpoints(paths):
        """
        Load each checkpoint from 'paths', sum their weights, and then average them.
        Return a model instance with the merged (averaged) weights.
        """
        if not paths:
            raise ValueError("No checkpoint paths provided for averaging.")

        model_tmp = ModelClass().to(device)
        state_dict_sums = None
        n = len(paths)

        for p in paths:
            cp = torch.load(p, map_location=device)
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

    # -----------------------------
    # 3) Incrementally merge top i checkpoints and track best
    # -----------------------------
    best_loss = math.inf
    best_state_dict = None
    best_combo_size = 0

    for i in range(1, len(top_ckpts) + 1):
        paths_to_merge = [ckpt_file for (ckpt_file, _) in top_ckpts[:i]]
        merged_model = average_checkpoints(paths_to_merge)
        test_loss_val = test(test_loader, merged_model, criterion, device)
        print(f"Ensemble of top {i} model(s) => test_loss={test_loss_val:.6f}")

        if test_loss_val < best_loss:
            best_loss = test_loss_val
            best_combo_size = i
            best_state_dict = merged_model.state_dict()

    # -----------------------------
    # 4) Save best merged model
    # -----------------------------
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
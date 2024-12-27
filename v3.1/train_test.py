from config import (
    TRAINING_PROGRESS_BAR,
    NUM_EPOCHS,
    CHECKPOINTS_DIR,
    MODEL_NAME,
    DEVICE,
    PROCESSED_DIR,
    PADDING,
    COARSE_RESOLUTION
)
import importlib
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Import our new metrics function
from metrics import metrics as compute_metrics
# Import tabulate for a nice table printout
from tabulate import tabulate

def train(train_dataloader, model, optimizer, criterion):
    """
    Train for one epoch. Data shape:
      'input':  (B, 2, fLat, fLon)
      'target': (B, fLat, fLon)
    """
    model.train()
    running_loss = 0.0
    total_steps = 0

    data_iter = tqdm(train_dataloader, desc="Training epoch") if TRAINING_PROGRESS_BAR else train_dataloader

    for batch in data_iter:
        inputs = batch['input'].float().to(DEVICE)    # (B, 2, fLat, fLon)
        targets = batch['target'].float().to(DEVICE)  # (B, fLat, fLon)
        optimizer.zero_grad()

        outputs = model(inputs)                       # (B, 1, fLat, fLon)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_steps += 1

    avg_loss = running_loss / total_steps if total_steps > 0 else 0.0
    return avg_loss


def test(test_dataloader, model, criterion):
    """
    Evaluate for one epoch. Also computes metrics vs. bilinear interpolation.
    
    Data shape is the same as train():
      'input':  (B, 2, fLat, fLon)  -> channel 0 is upsampled coarse precip (bilinear baseline)
      'target': (B, fLat, fLon)
    
    We accumulate model predictions, bilinear predictions, and targets
    for the entire test set. Then we compute MSE, MAE, bias, correlation in both
    normalized and unnormalized space, and display them in a table.
    """
    model.eval()
    running_loss = 0.0
    total_steps = 0

    # Lists for metrics
    model_preds = []
    bilinear_preds = []
    targets_all = []

    # Load normalization stats from combined_data.npz
    data_path = PROCESSED_DIR / "combined_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find {data_path}. Please run data_preprocessing.py first.")
    with np.load(data_path) as d:
        precip_mean = d["precip_mean"].item()
        precip_std = d["precip_std"].item()

    data_iter = tqdm(test_dataloader, desc="Testing epoch") if TRAINING_PROGRESS_BAR else test_dataloader

    with torch.no_grad():
        for batch in data_iter:
            inputs = batch['input'].float().to(DEVICE)    # (B, 2, fLat, fLon)
            targets = batch['target'].float().to(DEVICE)  # (B, fLat, fLon)

            outputs = model(inputs)                       # (B, 1, fLat, fLon)
            loss = criterion(outputs, targets.unsqueeze(1))
            running_loss += loss.item()
            total_steps += 1

            # coarse_input = batch['coarse_input'].float().to(DEVICE)  # (B, cLat, cLon)
            # pad_cells = int(round(PADDING / COARSE_RESOLUTION))
            # cropped_coarse_input = coarse_input[:, pad_cells:-pad_cells, pad_cells:-pad_cells]
            # interpolated_cropped_coarse_input = F.interpolate(
            #         cropped_coarse_input.unsqueeze(1),
            #         size=batch['target'].shape[-2:],  # (fLat, fLon)
            #         mode='bilinear',
            #         align_corners=False
            #     ).squeeze(1)

            # # Accumulate for metrics
            # model_preds.append(outputs.squeeze(1).cpu().numpy())     # shape (B, fLat, fLon)
            # bilinear_preds.append(interpolated_cropped_coarse_input.cpu().numpy())
            # targets_all.append(targets.cpu().numpy())

    avg_loss = running_loss / total_steps if total_steps > 0 else 0.0

    # # Concatenate along batch dimension
    # model_pred_all = np.concatenate(model_preds, axis=0)     # shape (N, fLat, fLon)
    # bilinear_pred_all = np.concatenate(bilinear_preds, axis=0)
    # target_all = np.concatenate(targets_all, axis=0)

    # # Compute metrics
    # metrics_dict = compute_metrics(
    #     model_pred_all,
    #     bilinear_pred_all,
    #     target_all,
    #     precip_mean,
    #     precip_std
    # )

    # # Build a table with rows = metrics, columns in the requested order:
    # #   [model normalized, bilinear normalized, model unnormalized, bilinear unnormalized]
    # table_data = []
    # metric_keys = ["mse", "mae", "bias", "corr"]
    # row_labels = ["MSE", "MAE", "Bias", "Corr"]

    # for mk, row_label in zip(metric_keys, row_labels):
    #     table_data.append([
    #         row_label,
    #         metrics_dict["model"][mk]["normalized"],
    #         metrics_dict["bilinear"][mk]["normalized"],
    #         metrics_dict["model"][mk]["unnormalized"],
    #         metrics_dict["bilinear"][mk]["unnormalized"]
    #     ])

    # headers = ["Metric", "Model (norm)", "Bilinear (norm)", "Model (unnorm)", "Bilinear (unnorm)"]

    # print("\n--- Detailed Metrics (Test) ---")
    # print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", floatfmt=".6f"))

    return avg_loss


def train_test(train_dataloader, test_dataloader):
    """
    Full train/test loop for NUM_EPOCHS.
    """
    print(f"Using device: {DEVICE}")

    # Dynamically load the model
    model_module = importlib.import_module(f"{MODEL_NAME}")
    ModelClass = model_module.Model
    model = ModelClass().to(DEVICE)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.SmoothL1Loss()

    # Run training & testing loops
    for epoch in range(0, NUM_EPOCHS):
        train_loss = train(train_dataloader, model, optimizer, criterion)
        test_loss = test(test_dataloader, model, criterion)

        # Save checkpoint
        checkpoint_path = CHECKPOINTS_DIR / f"{epoch}_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss
        }, checkpoint_path)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    print("Training complete!")
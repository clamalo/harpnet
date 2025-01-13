# File: /train_test.py

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

# We'll still import 'metrics' for reference, but we will NOT call it directly
# to avoid loading entire arrays into memory at once.
from metrics import metrics as reference_metrics
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
    Evaluate for one epoch using a streaming approach for metrics, to avoid
    excessive memory usage. Also compares vs. bilinear interpolation.

    Data shape is the same as train():
      'input':  (B, 2, fLat, fLon)  -> channel 0 is upsampled coarse precip (bilinear baseline)
      'target': (B, fLat, fLon)

    For each batch, we:
      1) Compute the model's output and accumulate loss for test_loss.
      2) Gather the bilinear baseline: we take 'coarse_input', crop padding, and bilinearly interpolate.
      3) Update partial sums (MSE, MAE, Bias, Corr) in both normalized and unnormalized space for
         (model vs. target) and (bilinear vs. target).

    At the end, we finalize the metrics from the partial sums, print them in a table,
    and return the average test loss.
    """

    model.eval()
    running_loss = 0.0
    total_steps = 0

    # We need normalization stats for unnormalizing
    data_path = PROCESSED_DIR / "combined_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find {data_path}. Please run data_preprocessing.py first.")

    with np.load(data_path) as d:
        precip_mean = d["precip_mean"].item()
        precip_std = d["precip_std"].item()

    # Helper accumulators for streaming metrics
    # We'll keep sums for MSE, MAE, Bias, Corr in normalized space (model vs. target)
    # and also for bilinear vs. target. Then for unnormalized space as well.
    class OnlineAccumulator:
        """
        A simple accumulator to track partial sums for MSE, MAE, Bias, and
        correlation in a streaming fashion.
        """
        def __init__(self):
            self.sum_x = 0.0         # sum of predictions
            self.sum_y = 0.0         # sum of targets
            self.sum_x_sq = 0.0      # sum of (predictions^2)
            self.sum_y_sq = 0.0      # sum of (targets^2)
            self.sum_xy = 0.0        # sum of (pred * target)
            self.sum_abs_diff = 0.0  # sum of absolute differences (for MAE)
            self.sum_sq_diff = 0.0   # sum of squared differences (for MSE)
            self.sum_diff = 0.0      # sum of (pred - target) (for bias)
            self.n = 0               # total number of elements

        def update(self, x, y):
            """
            Update partial sums with flat 1D numpy arrays x, y.
            """
            # Basic sums
            self.n += x.size
            self.sum_x += float(x.sum())
            self.sum_y += float(y.sum())
            self.sum_x_sq += float((x * x).sum())
            self.sum_y_sq += float((y * y).sum())
            self.sum_xy += float((x * y).sum())

            # MSE, MAE, bias
            diff = x - y
            self.sum_sq_diff += float((diff * diff).sum())   # for MSE
            self.sum_abs_diff += float(np.abs(diff).sum())   # for MAE
            self.sum_diff += float(diff.sum())               # for bias

        def finalize(self):
            """
            Compute final MSE, MAE, Bias, Corr from partial sums.
            """
            if self.n == 0:
                # Degenerate: no data
                return {
                    "mse": 0.0,
                    "mae": 0.0,
                    "bias": 0.0,
                    "corr": 0.0
                }

            mse_val = self.sum_sq_diff / self.n
            mae_val = self.sum_abs_diff / self.n
            bias_val = self.sum_diff / self.n

            # Correlation
            # r = [N * sum_xy - (sum_x)(sum_y)] / sqrt([N*sum_x_sq - (sum_x)^2]*[N*sum_y_sq - (sum_y)^2])
            numerator = (self.n * self.sum_xy) - (self.sum_x * self.sum_y)
            denom_x = (self.n * self.sum_x_sq) - (self.sum_x ** 2)
            denom_y = (self.n * self.sum_y_sq) - (self.sum_y ** 2)
            if denom_x <= 1e-12 or denom_y <= 1e-12:
                corr_val = 0.0
            else:
                corr_val = numerator / np.sqrt(denom_x * denom_y)

            return {
                "mse": mse_val,
                "mae": mae_val,
                "bias": bias_val,
                "corr": float(corr_val)
            }

    # We'll keep four accumulators: model_norm, bilinear_norm, model_unnorm, bilinear_unnorm
    model_norm_acc = OnlineAccumulator()
    bilinear_norm_acc = OnlineAccumulator()
    model_unnorm_acc = OnlineAccumulator()
    bilinear_unnorm_acc = OnlineAccumulator()

    data_iter = tqdm(test_dataloader, desc="Testing epoch") if TRAINING_PROGRESS_BAR else test_dataloader

    with torch.no_grad():
        for batch in data_iter:
            inputs = batch['input'].float().to(DEVICE)    # (B, 2, fLat, fLon)
            targets = batch['target'].float().to(DEVICE)  # (B, fLat, fLon)

            # Model outputs
            outputs = model(inputs)  # (B, 1, fLat, fLon)
            loss = criterion(outputs, targets.unsqueeze(1))
            running_loss += loss.item()
            total_steps += 1

            # Bilinear baseline: we have 'coarse_input' which is (B, cLat, cLon).
            # We remove the padding around the edges (PADDING / COARSE_RESOLUTION),
            # then upsample to fLat/fLon. That yields (B, fLat, fLon).
            coarse_input = batch['coarse_input'].float().to(DEVICE)
            pad_cells = int(round(PADDING / COARSE_RESOLUTION))
            # Crop out the padding in coarse_input
            if coarse_input.shape[-1] > 2 * pad_cells and coarse_input.shape[-2] > 2 * pad_cells:
                cropped_coarse_input = coarse_input[:, pad_cells:-pad_cells, pad_cells:-pad_cells]
            else:
                # Edge case: if for some reason pad_cells is large, skip crop
                cropped_coarse_input = coarse_input

            bilinear_pred = F.interpolate(
                cropped_coarse_input.unsqueeze(1),
                size=targets.shape[-2:],  # (fLat, fLon)
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # shape (B, fLat, fLon)

            # -------------------------------------------------
            # 1) Normalized-space accumulators
            # -------------------------------------------------
            # Model predictions are already in normalized log space (B, 1, fLat, fLon)
            model_pred_norm = outputs.squeeze(1).cpu().numpy()    # shape (B, fLat, fLon)
            bilinear_pred_norm = bilinear_pred.cpu().numpy()      # shape (B, fLat, fLon)
            target_norm = targets.cpu().numpy()                   # shape (B, fLat, fLon)

            # Flatten
            model_pred_norm_flat = model_pred_norm.reshape(-1)
            bilinear_pred_norm_flat = bilinear_pred_norm.reshape(-1)
            target_norm_flat = target_norm.reshape(-1)

            model_norm_acc.update(model_pred_norm_flat, target_norm_flat)
            bilinear_norm_acc.update(bilinear_pred_norm_flat, target_norm_flat)

            # -------------------------------------------------
            # 2) Unnormalized-space accumulators
            # -------------------------------------------------
            # Unnormalize from log space => mm
            # x_norm -> x_log = x_norm*std + mean => x_mm = expm1(x_log)
            # (We assume 'target_norm' is also in that same log/normalized domain)
            model_log = (model_pred_norm_flat * precip_std) + precip_mean
            model_mm = np.expm1(model_log)
            bilinear_log = (bilinear_pred_norm_flat * precip_std) + precip_mean
            bilinear_mm = np.expm1(bilinear_log)

            target_log = (target_norm_flat * precip_std) + precip_mean
            target_mm = np.expm1(target_log)

            model_unnorm_acc.update(model_mm, target_mm)
            bilinear_unnorm_acc.update(bilinear_mm, target_mm)

    # Average test loss
    avg_loss = running_loss / total_steps if total_steps > 0 else 0.0

    # -------------------------------------------------
    # Finalize streaming metrics for each
    # -------------------------------------------------
    model_norm_res = model_norm_acc.finalize()       # keys: mse, mae, bias, corr
    bilinear_norm_res = bilinear_norm_acc.finalize()
    model_unnorm_res = model_unnorm_acc.finalize()
    bilinear_unnorm_res = bilinear_unnorm_acc.finalize()

    # Build a dictionary that follows the same structure used by metrics.py
    metrics_dict = {
        "model": {
            "mse": {
                "normalized": float(model_norm_res["mse"]),
                "unnormalized": float(model_unnorm_res["mse"])
            },
            "mae": {
                "normalized": float(model_norm_res["mae"]),
                "unnormalized": float(model_unnorm_res["mae"])
            },
            "bias": {
                "normalized": float(model_norm_res["bias"]),
                "unnormalized": float(model_unnorm_res["bias"])
            },
            "corr": {
                "normalized": float(model_norm_res["corr"]),
                "unnormalized": float(model_unnorm_res["corr"])
            }
        },
        "bilinear": {
            "mse": {
                "normalized": float(bilinear_norm_res["mse"]),
                "unnormalized": float(bilinear_unnorm_res["mse"])
            },
            "mae": {
                "normalized": float(bilinear_norm_res["mae"]),
                "unnormalized": float(bilinear_unnorm_res["mae"])
            },
            "bias": {
                "normalized": float(bilinear_norm_res["bias"]),
                "unnormalized": float(bilinear_unnorm_res["bias"])
            },
            "corr": {
                "normalized": float(bilinear_norm_res["corr"]),
                "unnormalized": float(bilinear_unnorm_res["corr"])
            }
        }
    }

    # Format a table for printing
    metric_keys = ["mse", "mae", "bias", "corr"]
    row_labels = ["MSE", "MAE", "Bias", "Corr"]
    table_data = []

    for mk, row_label in zip(metric_keys, row_labels):
        table_data.append([
            row_label,
            metrics_dict["model"][mk]["normalized"],
            metrics_dict["bilinear"][mk]["normalized"],
            metrics_dict["model"][mk]["unnormalized"],
            metrics_dict["bilinear"][mk]["unnormalized"]
        ])

    headers = ["Metric", "Model (norm)", "Bilinear (norm)", "Model (unnorm)", "Bilinear (unnorm)"]

    print("\n--- Detailed Metrics (Test) [Streaming] ---")
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", floatfmt=".6f"))

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
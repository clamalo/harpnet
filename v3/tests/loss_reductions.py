"""
Script to analyze and print out loss reductions for ensemble models vs bilinear baseline.
"""

import torch
import math
from pathlib import Path

tiles = range(0, 36)
best_dir = Path("/Users/clamalo/documents/harpnet/v2/best")

rmse_reductions = []
mse_reductions = []
test_losses = []

for tile in tiles:
    ckpt_path = best_dir / f"{tile}_model.pt"
    if not ckpt_path.exists():
        print(f"No checkpoint found for tile {tile}. Skipping.")
        continue

    checkpoint = torch.load(ckpt_path, map_location=torch.device('mps'))
    test_loss = checkpoint['test_loss']
    bilinear_test_loss = checkpoint['bilinear_test_loss']
    rmse_reduction = 1 - (math.sqrt(test_loss) / math.sqrt(bilinear_test_loss))
    mse_reduction = 1 - (test_loss / bilinear_test_loss)
    rmse_reductions.append(rmse_reduction)
    mse_reductions.append(mse_reduction)
    test_losses.append(test_loss)
    print(f'Tile {tile} reduction: {(rmse_reduction*100):.2f}%')
    print(f'Test loss: {test_loss:.4f}')
    print(f'Bilinear test loss: {bilinear_test_loss:.4f}')
    print()

mean_rmse_reduction = sum(rmse_reductions) / len(rmse_reductions) if rmse_reductions else float('inf')
mean_mse_reduction = sum(mse_reductions) / len(mse_reductions) if mse_reductions else float('inf')
mean_test_loss = sum(test_losses) / len(test_losses) if test_losses else float('inf')

print(f'Mean RMSE reduction: {mean_rmse_reduction*100:.2f}%')
print(f'Mean MSE reduction: {mean_mse_reduction*100:.2f}%')
print(f'Mean test loss: {mean_test_loss:.4f}')
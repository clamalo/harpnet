import torch
import math

tiles = range(0, 16)

rmse_reductions = []
mse_reductions = []

for tile in tiles:
    checkpoint = torch.load(f'/Users/clamalo/downloads/{tile}_model.pt', map_location=torch.device('mps'))
    # checkpoint = torch.load(f'/Users/clamalo/documents/harpnet/best/{tile}_model.pt', map_location=torch.device('mps'))
    # checkpoint = torch.load(f'/Volumes/seagate/checkpoints/{tile}/19_model.pt', map_location=torch.device('mps'))
    test_loss = checkpoint['test_loss']
    bilinear_test_loss = checkpoint['bilinear_test_loss']
    rmse_reduction = 1 - (math.sqrt(test_loss) / math.sqrt(bilinear_test_loss))
    mse_reduction = 1 - (test_loss / bilinear_test_loss)
    rmse_reductions.append(rmse_reduction)
    mse_reductions.append(mse_reduction)
    print(f'Tile {tile} reduction: {(rmse_reduction*100):.2f}%')
    print(f'Test loss: {test_loss:.4f}')
    print(f'Bilinear test loss: {bilinear_test_loss:.4f}')
    print()
    print()

mean_rmse_reduction = sum(rmse_reductions) / len(rmse_reductions)
mean_mse_reduction = sum(mse_reductions) / len(mse_reductions)
print(f'Mean RMSE reduction: {mean_rmse_reduction*100:.2f}%')
print(f'Mean MSE reduction: {mean_mse_reduction*100:.2f}%')
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from src.generate_dataloaders import generate_dataloaders
import src.constants as constants
from src.model import UNetWithAttention
from src.get_coordinates import get_coordinates
from src.setup import setup
from src.xr_to_np import xr_to_np




start_month = (1979, 10)
end_month = (1980, 9)
train_test_ratio = 0.2
tile = 31

setup(tile)
# xr_to_np(tile, start_month, end_month)




model = UNetWithAttention(1, 1, output_shape=(64,64)).to(constants.torch_device)
checkpoint = torch.load(f"{tile}_model.pt", map_location=constants.torch_device)
model.load_state_dict(checkpoint['model_state_dict'])

coarse_lats_pad, coarse_lons_pad, coarse_lats, coarse_lons, fine_lats, fine_lons = get_coordinates(tile)

train_dataloader, test_dataloader = generate_dataloaders(tile, start_month, end_month, train_test_ratio)


losses = []
bilinear_losses = []
for (inputs, targets, times) in tqdm(test_dataloader, total=len(test_dataloader)):
    inputs, targets = inputs.to(constants.torch_device), targets.to(constants.torch_device)

    with torch.no_grad():
        outputs = model(inputs)

    loss = ((outputs - targets) ** 2).cpu().detach().numpy()

    cropped_inputs = inputs[:, 1:-1, 1:-1]
    interpolated_inputs = torch.nn.functional.interpolate(cropped_inputs.unsqueeze(1), size=(64, 64), mode='bilinear').squeeze(1)
    bilinear_loss = ((interpolated_inputs - targets) ** 2).cpu().detach().numpy()

    losses.append(loss.mean(axis=0))
    bilinear_losses.append(bilinear_loss.mean(axis=0))


mean_losses = np.array(losses).mean(axis=0)
mean_bilinear_losses = np.array(bilinear_losses).mean(axis=0)

max_value = max(mean_losses.max(), mean_bilinear_losses.max())

intermediate_lats = coarse_lats[:-1] + ((coarse_lats[1:] - coarse_lats[:-1]) / 2)
intermediate_lons = coarse_lons[:-1] + ((coarse_lons[1:] - coarse_lons[:-1]) / 2)




#plot the mean_losses with cartopy states 
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([fine_lons[0], fine_lons[-1], fine_lats[0], fine_lats[-1]])
ax.set_title('Mean Losses')
cf = ax.imshow(mean_losses, origin='upper', extent=(fine_lons[0], fine_lons[-1], fine_lats[0], fine_lats[-1]), transform=ccrs.PlateCarree(), vmin=0, vmax=max_value)
plt.colorbar(cf, ax=ax, orientation='horizontal', label='Mean Losses')
for lat in intermediate_lats:
    ax.axhline(y=lat, color='red', linestyle='--', linewidth=0.5)
for lon in intermediate_lons:
    ax.axvline(x=lon, color='red', linestyle='--', linewidth=0.5)
plt.savefig('model_losses.png')

#plot the mean_bilinear_losses with cartopy states 
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([fine_lons[0], fine_lons[-1], fine_lats[0], fine_lats[-1]])
ax.set_title('Mean Losses')
cf = ax.imshow(mean_bilinear_losses, origin='upper', extent=(fine_lons[0], fine_lons[-1], fine_lats[0], fine_lats[-1]), transform=ccrs.PlateCarree(), vmin=0, vmax=max_value)
plt.colorbar(cf, ax=ax, orientation='horizontal', label='Mean Losses')
for lat in intermediate_lats:
    ax.axhline(y=lat, color='red', linestyle='--', linewidth=0.5)
for lon in intermediate_lons:
    ax.axvline(x=lon, color='red', linestyle='--', linewidth=0.5)
plt.savefig('bilinear_losses.png')
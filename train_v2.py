import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime
from utils.model import UNetWithAttention
from utils.utils2 import *
import matplotlib.pyplot as plt
import cartopy
from metpy.plots import USCOUNTIES
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = '/Users/clamalo/documents/harpnet/load_data/'
domain = 14
ingest = True

# min_lat = get_lats_lons(domain)[0][0]
# max_lat = get_lats_lons(domain)[0][-1]
# min_lon = get_lats_lons(domain)[1][0]
# max_lon = get_lats_lons(domain)[1][-1]
# print(min_lat, max_lat, min_lon, max_lon)
# quit()

if ingest:
    create_grid_domains()
    xr_to_np(domain)

# check if domain checkpoints folder exists, if not create it
if not os.path.exists(f'checkpoints/{domain}'):
    os.makedirs(f'checkpoints/{domain}')

train_test_cutoff = '2020-10-01:00:00:00'
train_input_file_paths, train_target_file_paths, test_input_file_paths, test_target_file_paths = create_paths(train_test_cutoff, domain)

train_dataset, train_dataloader = create_dataloader(train_input_file_paths, train_target_file_paths)
test_dataset, test_dataloader = create_dataloader(test_input_file_paths, test_target_file_paths, shuffle=False)

print(len(train_dataloader), next(iter(train_dataloader))[0].numpy().shape, next(iter(train_dataloader))[1].numpy().shape)
print(len(test_dataloader), next(iter(test_dataloader))[0].numpy().shape, next(iter(test_dataloader))[1].numpy().shape)

model = UNetWithAttention(1, 1, output_shape=(64,64)).to('mps')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def test(model, dataloader, criterion, device):
    lats, lons, input_lats, input_lons = get_lats_lons(domain)
    model.eval()
    running_loss = 0.0
    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, targets)
        running_loss += loss.item()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': cartopy.crs.PlateCarree()})
        axs[0].pcolormesh(input_lons, input_lats, inputs[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=1)
        axs[1].pcolormesh(lons, lats, outputs[0].cpu().detach().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=1)
        axs[2].pcolormesh(lons, lats, targets[0].cpu().numpy(), transform=cartopy.crs.PlateCarree(), vmin=0, vmax=1)
        for ax in axs:
            ax.coastlines()
            ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray')
        box = patches.Rectangle((lons[0], lats[0]), lons[-1] - lons[0], lats[-1] - lats[0],
                                linewidth=1, edgecolor='r', facecolor='none')
        axs[0].add_patch(box)
        plt.savefig(f'figures/test/{i}.png')


    return running_loss / len(dataloader)

num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, 'mps')
    test_loss = test(model, test_dataloader, criterion, 'mps')
    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss
    }
    torch.save(checkpoint, f'checkpoints/{domain}/{epoch}_model.pt')
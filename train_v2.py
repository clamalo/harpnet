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


domain = 14
ingest = False
first_month = (1979, 10)
last_month = (2022, 9)
train_test_cutoff = '2020-10-01:00:00:00'


if ingest:
    setup(domain)
    create_grid_domains()
    xr_to_np(domain, first_month, last_month)

train_input_file_paths, train_target_file_paths, test_input_file_paths, test_target_file_paths = create_paths(domain, first_month, last_month, train_test_cutoff)

train_dataset, train_dataloader = create_dataloader(train_input_file_paths, train_target_file_paths)
test_dataset, test_dataloader = create_dataloader(test_input_file_paths, test_target_file_paths, shuffle=False)

print(len(train_dataloader), next(iter(train_dataloader))[0].numpy().shape, next(iter(train_dataloader))[1].numpy().shape)
print(len(test_dataloader), next(iter(test_dataloader))[0].numpy().shape, next(iter(test_dataloader))[1].numpy().shape)

model = UNetWithAttention(1, 1, output_shape=(64,64)).to('mps')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, 'mps')
    test_loss = test(domain, model, test_dataloader, criterion, 'mps')
    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss
    }
    torch.save(checkpoint, f'checkpoints/{domain}/{epoch}_model.pt')
import torch.nn as nn
import torch
import numpy as np
import os
os.system('ulimit -n 1024')
from utils.model import UNetWithAttention
from utils.utils import *
import matplotlib.pyplot as plt
from metpy.plots import USCOUNTIES
import warnings
warnings.filterwarnings("ignore")
import constants
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def test_bilinear(domain, model, dataloader, criterion, device, pad=False, plot=True):
    bilinear_losses = []

    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        
        cropped_inputs = inputs[:, 1:-1, 1:-1]
        interpolated_inputs = torch.nn.functional.interpolate(cropped_inputs.unsqueeze(1), size=(64, 64), mode='bilinear').squeeze(1)
        
        bilinear_loss = criterion(interpolated_inputs, targets)

        bilinear_losses.append(bilinear_loss.item())

    return np.mean(bilinear_losses)


domains = [1]

for domain in domains:
    print(domain)
    #check if corresponding domain directory exists
    if not os.path.exists(f'{constants.domains_dir}{domain}'):
        LOAD = True
    else:
        LOAD = False
    first_month = (1979, 10)
    last_month = (2022, 9)
    train_test = 0.2
    continue_epoch = False
    max_epoch = 1
    pad = True

    if LOAD:
        setup(domain)
        create_grid_domains()
        xr_to_np(domain, first_month, last_month, pad=pad)

    train_input_file_paths, train_target_file_paths, test_input_file_paths, test_target_file_paths = create_paths(domain, first_month, last_month, train_test)

    train_dataset, train_dataloader = create_dataloader(train_input_file_paths, train_target_file_paths, shuffle=True)
    test_dataset, test_dataloader = create_dataloader(test_input_file_paths, test_target_file_paths, shuffle=False)

    print(len(train_dataloader), next(iter(train_dataloader))[0].numpy().shape, next(iter(train_dataloader))[1].numpy().shape)
    print(len(test_dataloader), next(iter(test_dataloader))[0].numpy().shape, next(iter(test_dataloader))[1].numpy().shape)

    model = UNetWithAttention(1, 1, output_shape=(64,64)).to('mps')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    if continue_epoch:
        checkpoint = torch.load(f'{constants.checkpoints_dir}{domain}/{continue_epoch-1}_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(continue_epoch or 0, max_epoch):
        # train_loss = train(domain, model, train_dataloader, criterion, optimizer, 'mps', pad=pad, plot=False)
        bilinear_loss = test_bilinear(domain, model, test_dataloader, criterion, 'mps', pad=pad, plot=False)
        train_loss, test_loss = 100, 100
        print(f'Epoch {epoch} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Bilinear Loss: {bilinear_loss:.4f}')
        
        #load each model checkpoint
        os.makedirs(f'{constants.checkpoints_dir}{domain}', exist_ok=True)
        for checkpoint_file in os.listdir(f'{constants.checkpoints_dir}{domain}'):
            checkpoint = torch.load(f'{constants.checkpoints_dir}{domain}/{checkpoint_file}')
            print(checkpoint['bilinear_loss'])
            checkpoint['bilinear_loss'] = bilinear_loss
            print(checkpoint['bilinear_loss'])
            print(checkpoint['test_loss'])
            print()
            torch.save(checkpoint, f'{constants.checkpoints_dir}{domain}/{checkpoint_file}')
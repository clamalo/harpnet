import torch.nn as nn
import torch
import numpy as np
import os
os.system('ulimit -n 1024')
import sys
import warnings
warnings.filterwarnings("ignore")
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cwd_dir = (os.path.abspath(os.path.join(os.getcwd())))
sys.path.insert(0, cwd_dir)
from utils.model import UNetWithAttention
from utils.utils3 import *
import utils.constants as constants

# 0 to 35
domains = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
domains = [12]

setup()
for domain in domains:
    LOAD = True
    first_month = (1979, 10)
    last_month = (1980, 9)
    train_test = 0.2
    continue_epoch = False
    max_epoch = 1
    pad = True

    if LOAD:
        setup(domain)
        create_grid_domains()
        xr_to_np(domain, first_month, last_month, pad=pad)

    quit()

    def generate_dataloaders(domain, first_month, last_month, train_test):
        input_file_paths = []
        target_file_paths = []
        times_file_paths = []
        first_month = datetime(first_month[0], first_month[1], 1)
        last_month = datetime(last_month[0], last_month[1], 1)
        current_month = first_month
        while current_month <= last_month:
            input_fp = f'{constants.domains_dir}{domain}/input_{current_month.year}_{current_month.month:02d}.npy'
            target_fp = f'{constants.domains_dir}{domain}/target_{current_month.year}_{current_month.month:02d}.npy'
            times_fp = f'{constants.domains_dir}{domain}/times_{current_month.year}_{current_month.month:02d}.npy'
            input_file_paths.append(input_fp)
            target_file_paths.append(target_fp)
            times_file_paths.append(times_fp)
            current_month += relativedelta(months=1)

        input_arr = np.concatenate([np.load(fp) for fp in input_file_paths])
        target_arr = np.concatenate([np.load(fp) for fp in target_file_paths])
        times_arr = np.concatenate([np.load(fp) for fp in times_file_paths])

        times_arr = times_arr.astype('datetime64[s]').astype(np.float64)

        np.random.seed(42)
        indices = np.argsort(times_arr)
        np.random.shuffle(indices)
        input_arr = input_arr[indices]
        target_arr = target_arr[indices]
        times_arr = times_arr[indices]

        print(times_arr[:3].astype('datetime64[s]'))

        test_input_arr, train_input_arr = np.split(input_arr, [int(train_test * len(input_arr))])
        test_target_arr, train_target_arr = np.split(target_arr, [int(train_test * len(target_arr))])
        test_times_arr, train_times_arr = np.split(times_arr, [int(train_test * len(times_arr))])

        train_dataset = MemMapDataset(train_input_arr, train_target_arr, train_times_arr)
        test_dataset = MemMapDataset(test_input_arr, test_target_arr, test_times_arr)

        torch.manual_seed(42)
        generator = torch.Generator()
        generator.manual_seed(42)
        train_dataloader = DataLoader(train_dataset, batch_size=constants.training_batch_size, shuffle=True, generator=generator)
        test_dataloader = DataLoader(test_dataset, batch_size=constants.training_batch_size, shuffle=False)

        return train_dataloader, test_dataloader

    train_dataloader, test_dataloader = generate_dataloaders(domain, first_month, last_month, train_test)

    print(len(train_dataloader), next(iter(train_dataloader))[0].numpy().shape, next(iter(train_dataloader))[1].numpy().shape)
    print(len(test_dataloader), next(iter(test_dataloader))[0].numpy().shape, next(iter(test_dataloader))[1].numpy().shape)

    model = UNetWithAttention(1, 1, output_shape=(64,64)).to(constants.device)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    if continue_epoch:
        checkpoint = torch.load(f'{constants.checkpoints_dir}{domain}/{continue_epoch-1}_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(continue_epoch or 0, max_epoch):
        train_loss = train(domain, model, train_dataloader, criterion, optimizer, constants.device, pad=pad, plot=False)
        test_loss, bilinear_loss = test(domain, model, test_dataloader, criterion, constants.device, pad=pad, plot=False)
        print(f'Epoch {epoch} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Bilinear Loss: {bilinear_loss:.4f}')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'bilinear_loss': bilinear_loss
        }
        #make sure directory exists
        os.makedirs(f'{constants.checkpoints_dir}{domain}', exist_ok=True)
        torch.save(checkpoint, f'{constants.checkpoints_dir}{domain}/{epoch}_model.pt')
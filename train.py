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
from utils.utils import *
import utils.constants as constants


domains = [2]


setup()
for domain in domains:
    LOAD = True
    first_month = (1979, 10)
    last_month = (1980, 9)
    train_test = 0.2
    continue_epoch = False
    max_epoch = 20
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

    model = UNetWithAttention(1, 1, output_shape=(64,64)).to(constants.device)
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